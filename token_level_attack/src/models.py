from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForTokenClassification, TableQuestionAnsweringPipeline
import torch
from src.utils import load_gpt2_from_dict
import torch.nn.functional as F

def bert_score(refs, cands, weights=None):
    refs_norm = refs / refs.norm(2, -1).unsqueeze(-1)
    if weights is not None:
        refs_norm *= weights[:, None]
    else:
        refs_norm /= refs.size(1)
    cands_norm = cands / cands.norm(2, -1).unsqueeze(-1)
    cosines = refs_norm @ cands_norm.transpose(1, 2)
    # remove first and last tokens; only works when refs and cands all have equal length (!!!)
    cosines = cosines[:, 1:-1, 1:-1]
    R = cosines.max(-1)[0].sum(1)
    return R

def get_model(model_name, num_labels, checkpoint_folder):
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_prefix_space=True
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
    ).cuda()

    model.load_state_dict(torch.load(checkpoint_folder))

    # if model_name == 'gpt2':
    #     tokenizer.padding_side = "right"
    #     tokenizer.pad_token = tokenizer.eos_token
    #     model.config.pad_token_id = model.config.eos_token_id

    return tokenizer, model

def load_ref_model(model_name):
    if 'bert-base-uncase' in model_name:
        ref_model = load_gpt2_from_dict("model/transformer_wikitext-103.pth", output_hidden_states=True).cuda()
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).cuda()
    return ref_model

def init_log_coeffs(input_ids, embeddings, args):
    with torch.no_grad():
        log_coeffs = torch.zeros(len(input_ids), embeddings.size(0))
        indices = torch.arange(log_coeffs.size(0)).long()
        log_coeffs[indices, torch.LongTensor(input_ids)] = args.initial_coeff
        log_coeffs = log_coeffs.cuda()
        log_coeffs.requires_grad = True
    return log_coeffs

def get_gumbel_softmax_embedding(raw_input, suffix_id_list, tau):
    gumbel_input = raw_input[:, :, suffix_id_list] if suffix_id_list is not None else raw_input
    sub_coeffs = F.gumbel_softmax(gumbel_input, hard=False, tau=tau)
    if suffix_id_list is not None:
        result_coeffs = raw_input.new_zeros(raw_input.shape)
        result_coeffs[:, :, suffix_id_list] = sub_coeffs
        return result_coeffs
    else:
        return sub_coeffs
def get_original_visual_embedding(original_subtoken_ids,suffix_visual_tensor,suffix_id_list):
    result=[]
    for id in original_subtoken_ids:
        index_in_matrix=suffix_id_list.index(id)
        result.append(suffix_visual_tensor[index_in_matrix][:])
    result=torch.stack(result,dim=0)
    return result
def get_adversial_loss(input_ids,entity_info_list,loss_type, log_coeffs, available_indices, origin_labels, labels, label_mask, num_labels, model,embeddings,entity_start_ends,token_type_ids_batch, args, suffix_id_list=None,suffix_list=None,suffix_visual_tensor=None):
    ### 这里修改一下逻辑，只对 mask 掉的词进行 gumbel softmax 采样，其他的词统一进行, 先通过list采样
    available_indices = available_indices.tolist()
    unavailable_indices = [i for i in range(len(input_ids)) if i not in available_indices]

    log_coeffs_input = log_coeffs.unsqueeze(0).repeat(args.batch_size, 1, 1)
    log_coeffs_input_hard=log_coeffs.unsqueeze(0).repeat(args.batch_size, 1, 1)
    # make sure coefficient is 1
    log_coeffs_input[:,unavailable_indices,:]/=args.initial_coeff
    log_coeffs_input_hard[:,unavailable_indices,:]/=args.initial_coeff

    sub_coeffs = get_gumbel_softmax_embedding(log_coeffs_input[:, available_indices, :], suffix_id_list, args.tau)
    sub_coeffs_view = sub_coeffs.view(-1, sub_coeffs.shape[2])
    new_sub_coeffs = sub_coeffs.new_zeros(sub_coeffs_view.shape)
    max_logits, idx = torch.max(sub_coeffs_view, dim=-1)

    new_sub_coeffs[torch.arange(0, len(idx)), idx] = max_logits * (1/max_logits.detach())
    new_sub_coeffs = new_sub_coeffs.view(sub_coeffs.shape)
    # sub_coeffs = F.gumbel_softmax(log_coeffs_input[:, available_indices, :], hard=False, tau=args.tau)  # B x T x V
    log_coeffs_input[:, available_indices, :] = sub_coeffs
    log_coeffs_input_hard[:, available_indices, :] =new_sub_coeffs
    inputs_embeds = (log_coeffs_input @ embeddings[None, :, :])  # B x T x D
    inputs_embeds_hard=log_coeffs_input_hard @ embeddings[None, :, :]
    
    total_loss, vis_loss, length_loss = 0, 0, 0
    if loss_type == 'direct':
        labels = torch.LongTensor([labels] * args.batch_size).cuda().view(-1)
        model_output = model(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids_batch, labels=labels)
        adv_loss = model_output.loss
        pred = model_output.logits

    elif loss_type == 'cw':
        def get_loss(inputs_embeds):

            labels_list = torch.LongTensor([origin_labels] * args.batch_size).cuda().view(-1)
            new_label_mask = [0]*len(origin_labels)
            # entity_start_pos is entity's start subtoken position
            for entity_info in entity_info_list:
                entity_start=entity_info['entity_begin_end'][0]
                new_label_mask[entity_start]=1
            masks = torch.BoolTensor([new_label_mask] * args.batch_size).cuda().view(-1)
            model_output = model(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids_batch)
            pred = model_output.logits
            # changable position output entity_nums*num_labels
            new_pred = torch.masked_select(pred.view(-1,  num_labels), masks.unsqueeze(dim=1)).view(-1,  num_labels)
            # entity_nums
            labels_list = torch.masked_select(labels_list, masks)
            # indices
            top_preds = new_pred.sort(descending=True)[1]
            correct = (top_preds[:, 0] == labels_list).long()
            indices = top_preds.gather(1, correct.view(-1, 1))
            labels_list=labels_list.view(-1,1)
            adv_loss = (new_pred.gather(1,labels_list) - new_pred.gather(1, indices) + args.kappa).clamp(min=0).mean()
            return adv_loss
        adv_loss =get_loss(inputs_embeds)
        adv_loss1 =get_loss(inputs_embeds_hard)
        total_loss += adv_loss
        total_loss += adv_loss1
    elif loss_type == 'ce':
        raise Exception("not implement")
    if args.visual_loss or args.length_loss:
        def use_length():
            # suffix_size
            suffix_lengths=torch.tensor([len(suffix)-2 for suffix in suffix_list], dtype=torch.float32 ,device='cuda')
            # B*mid_token_num*suffix_size
            target_suffix_distribution=sub_coeffs[:,:,suffix_id_list]
            # B*mid_token_num
            cur_average_length=target_suffix_distribution@suffix_lengths
            original_subtoken_ids=[input_ids[available_id] for available_id in available_indices]
            origin_lengths=[]
            for origin_subtoken_id in original_subtoken_ids:
                id_in_list=suffix_id_list.index(origin_subtoken_id)
                origin_lengths.append(suffix_lengths[id_in_list])
            origin_lengths=torch.tensor(origin_lengths,device='cuda').unsqueeze(0)
            loss=torch.abs(cur_average_length-origin_lengths).sum()
            loss= loss/(cur_average_length.size(0)*cur_average_length.size(1))
            return loss
        def use_visual_vec():
            # B*mid_token_num*suffix_size
            target_suffix_distribution=sub_coeffs[:,:,suffix_id_list]
            # out B * mid_token_num*2048
            cur_visual_embedding=target_suffix_distribution@suffix_visual_tensor
            original_subtoken_ids=[input_ids[available_id] for available_id in available_indices]
            # mid_token_num*2048
            original_visual_embedding=get_original_visual_embedding(original_subtoken_ids,suffix_visual_tensor,suffix_id_list)
            # broadcast
            # loss=(1-torch.cosine_similarity(cur_visual_embedding,original_visual_embedding,dim=-1)).sum()
            loss=torch.norm(cur_visual_embedding-original_visual_embedding,dim=-1).sum()
            loss=loss/(target_suffix_distribution.size(0)*target_suffix_distribution.size(1))
            return loss
        
        if args.length_loss:
            length_loss = use_length()
            total_loss+= args.lam_length*length_loss
        if args.visual_loss:
            vis_loss = use_visual_vec()
            total_loss += args.lam_visual*vis_loss 
    return total_loss, adv_loss, adv_loss1, vis_loss, length_loss

def get_similarity_loss(loss_type, coeffs, ref_embeddings, orig_output, ref_model, ref_weights, args):
    if args.lam_sim <= 0 or args.only_adv:
        return torch.Tensor([0]).cuda(), None
        
    ref_embeds = (coeffs @ ref_embeddings[None, :, :])
    pred = ref_model(inputs_embeds=ref_embeds)

    output = pred.hidden_states[args.embed_layer]
    if loss_type.startswith('bertscore'):
        ref_loss = -args.lam_sim * bert_score(orig_output, output, weights=ref_weights).mean()
    else:
        if args.model == 'gpt2' or 'bert-base-uncased' in args.model:
            output = output[:, -1]
        else:
            output = output.mean(1)
        cosine = (output * orig_output).sum(1) / output.norm(2, 1) / orig_output.norm(2, 1)
        ref_loss = -args.lam_sim * cosine.mean()
    return ref_loss, pred

def get_perplexity_loss(coeffs, pred, args):
    if args.lam_perp <= 0 or args.only_adv:
        return torch.Tensor([0]).cuda()
    logits = pred.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    loss = -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()
    return args.lam_perp * loss
    
    