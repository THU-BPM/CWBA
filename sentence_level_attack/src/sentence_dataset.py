import imp
import os
from selectors import EpollSelector
from datasets import load_dataset
from matplotlib.style import available
from src.attack.attack_data import entity_around_insert, entity_around_change, context_change, entity_char_attack, token_select_attack
from src.models import get_gumbel_softmax_embedding
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def load_data(dataset_name, num_samples, start_index, model_name):
    if model_name == 'bert-base-uncased':
        data_files = {
            'validation': os.path.join('data', dataset_name, 'test_processed2.json')
        }
    elif model_name == 'roberta-base':
        data_files = {
            'validation': os.path.join('data', dataset_name, 'test_processed_roberta.json')
        }
    elif model_name == 'albert-base-v1':
        data_files = {
            'validation': os.path.join('data', dataset_name, 'test_processed_albert.json')
        }
    elif model_name == 'xlnet-base-cased':
        data_files = {
            'validation': os.path.join('data', dataset_name, 'test_processed_xlnet.json')
        }
    elif model_name == 'gpt2-medium':
        data_files = {
            'validation': os.path.join('data', dataset_name, 'test_processed_gpt2.json')
        }
    with open(os.path.join('data', dataset_name, 'id2label.json'), 'r') as f:
        id2label = json.load(f)
    datasets = load_dataset('json', data_files=data_files)
    datasets['validation'] = datasets['validation'].select(range(start_index, start_index + num_samples))
    num_labels = 0
    if dataset_name == 'dbpedia':
        num_labels = 14
    elif dataset_name == 'ag_news':
        num_labels = 4
    elif dataset_name == 'imdb':
        num_labels = 2
    elif dataset_name == 'yelp':
        num_labels = 2

    return datasets, num_labels, id2label


def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        padding=False,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                cur_label=label[word_idx]
                label_ids.append(label_to_id[cur_label])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    # tokenized_inputs['ner_tags']=examples['ner_tags']
    return tokenized_inputs

def add_prefix(example):
    example['sentence'] = 'My sentence: ' + example['sentence']
    return example

def parse_number(insert_command):
    left = int(insert_command.split('_')[1][1:])
    right = int(insert_command.split('_')[2][1:])
    return left, right

# input a datasetdict and output a datasetdict
# use dataset.map method to add delete and revise
def adapt_dataset(mode, encoded_dataset, tokenizer, suffix_id_list, args):

    if mode == 'random':
        encoded_dataset['validation'] = encoded_dataset['validation'].map(token_select_attack, fn_kwargs={'tokenizer': tokenizer,'suffix_id_list':suffix_id_list, 'mode': 'random', 'args': args}, remove_columns=encoded_dataset['validation'].column_names, batched=True, load_from_cache_file =False)
        pass
    elif mode == 'grad':
        encoded_dataset['validation'] = encoded_dataset['validation'].map(token_select_attack, fn_kwargs={'tokenizer': tokenizer,'suffix_id_list':suffix_id_list, 'mode': 'grad',  'args': args}, remove_columns=encoded_dataset['validation'].column_names, batched=True, load_from_cache_file =False)
    elif mode == 'context':
        encoded_dataset['train'].map(context_change)
        encoded_dataset['validation'].map(context_change)
    elif mode == 'char_entity':
        # encoded_dataset['train'] = encoded_dataset['train'].map(entity_char_attack, fn_kwargs={'label_to_id': label_to_id, 'tokenizer': tokenizer,'suffix_id_list':suffix_id_list}, remove_columns=encoded_dataset['train'].column_names, batched=True, load_from_cache_file = False)
        encoded_dataset['validation'] = encoded_dataset['validation'].map(entity_char_attack, fn_kwargs={'label_to_id': label_to_id, 'tokenizer': tokenizer,'suffix_id_list':suffix_id_list}, remove_columns=encoded_dataset['validation'].column_names, batched=True, load_from_cache_file =False)
        pass
    elif mode == 'char_context':
        pass
    
    return encoded_dataset

def check_next_token(tokenizer, token_ids, select_index, args):
    if not args.ind_ent:
        return True
    after_token = tokenizer.decode(token_ids[select_index[-1] + 1])
    before_token = tokenizer.decode(token_ids[select_index[0] - 1])

    if after_token.startswith('#') or before_token.endswith('#'):
        return False
    elif after_token.endswith('#') or before_token.startswith('#'):
        return False
    else:
        return True

def get_pred(text, tokenizer, model):
    x = tokenizer(text, max_length=256, truncation=True, return_tensors='pt')
    logit = model(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda(), token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None)).logits.data.cpu()
    return logit.argmax(dim=1)[0]

def get_pred_from_ids(ids, model):
    # logit = model(input_ids=torch.cat((torch.tensor([101]), ids, torch.tensor([102])), dim=0).unsqueeze(0).cuda()).logits.data.cpu()
    logit = model(input_ids=ids.unsqueeze(0).cuda()).logits.data.cpu()
    return logit.argmax(dim=1)[0]

def get_soft_preds(available_indices, batch_size, log_coeffs, suffix_id_list, tau, embeddings, model):
    available_indices = available_indices.tolist()
    log_coeffs_input = log_coeffs.unsqueeze(0).repeat(batch_size, 1, 1)
    sub_coeffs = get_gumbel_softmax_embedding(log_coeffs_input[:, available_indices, :], suffix_id_list, tau)
    # sub_coeffs = F.gumbel_softmax(log_coeffs_input[:, available_indices, :], hard=False, tau=args.tau)  # B x T x V
    log_coeffs_input[:, available_indices, :] = sub_coeffs
    inputs_embeds = (log_coeffs_input @ embeddings[None, :, :])  # B x T x D
    logit = model(inputs_embeds=inputs_embeds).logits.data.cpu()
    return logit.argmax(dim=1)[0]

def get_soft_preds2(available_indices, batch_size, log_coeffs, suffix_id_list, tau, embeddings, model):
    available_indices = available_indices.tolist()
    log_coeffs_input = log_coeffs.unsqueeze(0).repeat(batch_size, 1, 1)
    sub_coeffs = get_gumbel_softmax_embedding(log_coeffs_input[:, available_indices, :], suffix_id_list, tau)
    # sub_coeffs = F.gumbel_softmax(log_coeffs_input[:, available_indices, :], hard=False, tau=args.tau)  # B x T x V
    sub_coeffs_view = sub_coeffs.view(-1, sub_coeffs.shape[2])
    new_sub_coeffs = sub_coeffs.new_zeros(sub_coeffs_view.shape)
    max_logits, idx = torch.max(sub_coeffs_view, dim=1)
    new_sub_coeffs[torch.arange(0, len(idx)), idx] = max_logits * (1/max_logits.detach())
    new_sub_coeffs = new_sub_coeffs.view(sub_coeffs.shape)
    log_coeffs_input[:, available_indices, :] = new_sub_coeffs
    ids = log_coeffs_input.argmax(2)
    label = get_pred_from_ids(ids.cpu().view(-1), model)
    return label, ids


def get_index_select(label_mask):
    index_select = []
    for index in range(len(label_mask)):
        if label_mask[index] == 0:
            index_select.append(index) 
    return index_select

def get_attack_entity_name(entity_start_ends, input_ids, tokenizer):
    entitys=[]
    for begin,end in entity_start_ends:
        entitys.append(tokenizer.decode([input_ids[id] for id in range(begin,end)]))
    return entitys

def gumbel_test(tokenizer, word):
    word_id = tokenizer.encode(word, add_special_tokens=False)
    log_coeffs = torch.zeros(len(word_id), 30522)
    indices = torch.arange(log_coeffs.size(0)).long()
    log_coeffs[indices, torch.LongTensor(word_id)] = 15
    sub_adv_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1)
    print(tokenizer.decode(sub_adv_ids))

def calculate_edit_distance(s1,s2):
    s1 = s1.replace(" ", "")
    s2 = s2.replace(" ", "")
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
    
def get_changable_index(begin,end):
    num_ids=end-begin
    if num_ids==2:
        return range(begin+1,begin+2)
    else:
        return range(begin+1,end-1)


def generate_adversial_text(input_ids, label, clean_text, log_coeffs, available_indices, tokenizer, model, args, embeddings, suffix_id_list=None, suffix_visual_tensor=None,offset=1):
    adv_text, success_attack, _adv_ids, attack_number = "", False, [], args.gumbel_samples
    mismatch=False
    sub_adv_ids_list = []
    with torch.no_grad():
        origin_log_coeffs = log_coeffs
        log_coeffs=log_coeffs.cpu()
        adv_ids = log_coeffs.argmax(1)
        attack_num = 0
        while True:
            adv_ids = log_coeffs.argmax(1)
            suffix_id_list=torch.LongTensor(suffix_id_list)
            suffix_candidate_weights=log_coeffs[available_indices][:,suffix_id_list]
            # import ipdb; ipdb.set_trace()
            assert len(suffix_candidate_weights)!=0
            sub_adv_ids = F.gumbel_softmax(suffix_candidate_weights, hard=True, tau=args.tau).argmax(1)
            adv_ids[available_indices] = suffix_id_list[sub_adv_ids]
            # if sub_adv_ids.tolist() in sub_adv_ids_list:
            #     continue
            attack_num += 1
            sub_adv_ids_list.append(sub_adv_ids.tolist())
            _adv_ids = adv_ids
            
            # adv_text = tokenizer.decode(_adv_ids)
            texts = tokenizer.convert_ids_to_tokens(_adv_ids)
            if args.model == 'bert-base-uncased' or args.model == 'albert-base-v1' or args.model == 'xlnet-base-cased':
                adv_text = tokenizer.decode(_adv_ids)
            elif args.model == 'roberta-base' or args.model == 'gpt2-medium':
                result_text = ""
                for text in texts:
                    result_text += text.replace('Ġ', ' ')
                adv_text = tokenizer.decode(tokenizer.encode(result_text)[1:-1])

            soft_preds = get_soft_preds(available_indices, args.batch_size, origin_log_coeffs, suffix_id_list, args.tau, embeddings, model)
            soft_preds2, ids = get_soft_preds2(available_indices, args.batch_size, origin_log_coeffs, suffix_id_list, args.tau, embeddings, model)
            # import ipdb; ipdb.set_trace()
            # adv_text = tokenizer.decode(_adv_ids)
            origin_preds = get_pred_from_ids(_adv_ids, model)
            adv_preds = get_pred(adv_text, tokenizer, model)
            # import ipdb; ipdb.set_trace()
            # if tokenizer.encode(adv_text)[1:-1] !=_adv_ids.tolist():
            #     import ipdb; ipdb.set_trace()
            # if adv_preds!= origin_preds:
            # import ipdb; ipdb.set_trace()
            if (adv_preds.tolist()!=label):
                success_attack = True
                attack_number = attack_num
                print('attack success')
                break
            if attack_num >= args.gumbel_samples:
                break
        
    print(soft_preds, soft_preds2, origin_preds, adv_preds)
    print(adv_text)
    # import ipdb; ipdb.set_trace()
    return adv_text, success_attack, _adv_ids, attack_number, adv_preds.tolist()
    
def check_best_visual(label_mask, input_ids, label, clean_text, log_coeffs, available_indices, tokenizer, model,args, embeddings, suffix_id_list=None,suffix_visual_tensor=None,offset=1):
    adv_text, success_attack = "", False
    mismatch=False
    index_select = get_index_select(label_mask)
    if len(available_indices)>1:
        return adv_text, success_attack,mismatch,1
    with torch.no_grad():
        adv_ids = log_coeffs.argmax(1)
        log_coeffs=log_coeffs.cpu()
        origin_id=input_ids[index_select[1]]
        suffix_id_list=torch.LongTensor(suffix_id_list)
        suffix_ids=suffix_id_list.tolist()
        origin_id_in_mat=suffix_ids.index(origin_id)
        origin_tensor=suffix_visual_tensor[origin_id_in_mat]
        sim_rank=[]
        for i,candidate_tensor in enumerate(suffix_visual_tensor):
            if i==origin_id_in_mat: continue
            sim_rank.append((i,torch.norm(origin_tensor-candidate_tensor)))
        sim_rank=sorted(sim_rank,key=lambda x: x[1])
        res=[]
        res.append(tokenizer.convert_ids_to_tokens(origin_id))
        for id_and_sim in sim_rank:
            cur_id=id_and_sim[0]
            sim=id_and_sim[1]
            cur_token=tokenizer.convert_ids_to_tokens(suffix_ids[cur_id])
            adv_ids[available_indices[0]]=suffix_id_list[cur_id].cuda()
            _adv_ids = adv_ids[offset : len(adv_ids) - offset]
            adv_text = tokenizer.decode(_adv_ids)
            adv_preds = get_pred(adv_text, tokenizer, model)
            entity_label=label[index_select[0]]
            adv_label=adv_preds[index_select[0]]
            if (entity_label!=adv_label and 
                    check_next_token(tokenizer, adv_ids, index_select, args)):

                new_entity_str=tokenizer.decode(adv_ids[index_select])
                # re encode the new entity str but different from desire entity ids
                if tokenizer.encode(new_entity_str,add_special_tokens=False)!=adv_ids[index_select].tolist():
                    mismatch=True
                success_attack = True
                ele=(cur_token,sim)
                res.append(ele)
                if len(res)>=6:
                    print(res)
                    break
    return adv_text, success_attack,mismatch,1
          
        
    # check all candidate success rate
    # with torch.no_grad():
    #     log_coeffs=log_coeffs.cpu()
    #     adv_ids = log_coeffs.argmax(1)
    #     success=0
    #     V=len(suffix_id_list)
    #     for id in suffix_id_list:
    #         adv_ids[available_indices]=torch.LongTensor([id])
    #         _adv_ids = adv_ids[offset : len(adv_ids) - offset]
    #         adv_text = tokenizer.decode(_adv_ids)
    #         adv_preds = get_pred(adv_text, tokenizer, model)
    #         entity_label=label[index_select[0]]
    #         adv_label=adv_preds[index_select[0]]
    #         if (entity_label!=adv_label):
    #             success+=1
    #     print(success)
    #     print(success/V)

    # check weight distribution
            # for tsor in suffix_candidate_weights:
            #     tsor=tsor.cpu()
            #     exp_tsor=torch.exp(tsor)
            #     exp_tsor/=torch.sum(exp_tsor)
            #     print(exp_tsor.sum())
            #     plt.cla()
            #     plt.scatter(torch.arange(len(tsor)),exp_tsor)
            #     plt.savefig("tmp")
    # check sampled vec's similarity with original one
            # for avid,new_id in zip(available_indices,sub_adv_ids):
            #     token_id=input_ids[avid]
            #     id_in_mat=suffix_id_list.tolist().index(token_id)
            #     origin_vis_tensor=suffix_visual_tensor[id_in_mat]
            #     changed_vis_tensor=suffix_visual_tensor[new_id]
            #     dis=torch.norm(origin_vis_tensor-changed_vis_tensor)
            #     print(dis)
            #     print(F.cosine_similarity(origin_vis_tensor,changed_vis_tensor,dim=0))
            # clean_preds = get_pred(clean_text, tokenizer, model)
    