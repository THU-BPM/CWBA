import os
from datasets import load_dataset
from src.attack.attack_data import entity_around_insert, entity_around_change, context_change, entity_char_attack
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from src.utils import get_pure_input_ids
from copy import deepcopy
def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        padding=False,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i,label in enumerate(examples['ner_tags']):
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
def adapt_dataset(mode, encoded_dataset, label_to_id, tokenizer,suffix_id_list):

    if mode.startswith('insert'):
        left, right = parse_number(mode)
        encoded_dataset['train'] = encoded_dataset['train'].map(entity_around_insert, fn_kwargs={'label_to_id': label_to_id, 'left_number': left, 'right_number': right}, remove_columns=encoded_dataset['train'].column_names, batched=True, load_from_cache_file = False)
        encoded_dataset['validation'] = encoded_dataset['validation'].map(entity_around_insert, fn_kwargs={'label_to_id': label_to_id, 'left_number': left, 'right_number': right}, remove_columns=encoded_dataset['validation'].column_names, batched=True, load_from_cache_file =False)
    elif mode.startswith('nearby'):
        left, right = parse_number(mode)
        encoded_dataset['train'] = encoded_dataset['train'].map(entity_around_change, fn_kwargs={'label_to_id': label_to_id, 'left_number': left, 'right_number': right}, remove_columns=encoded_dataset['train'].column_names, batched=True, load_from_cache_file = False)
        encoded_dataset['validation'] = encoded_dataset['validation'].map(entity_around_change, fn_kwargs={'label_to_id': label_to_id, 'left_number': left, 'right_number': right}, remove_columns=encoded_dataset['validation'].column_names, batched=True, load_from_cache_file =False)
    elif mode == 'context':
        encoded_dataset['train'].map(context_change)
        encoded_dataset['validation'].map(context_change)
    elif mode == 'char_entity':
        # encoded_dataset['train'] = encoded_dataset['train'].map(entity_char_attack, fn_kwargs={'label_to_id': label_to_id, 'tokenizer': tokenizer,'suffix_id_list':suffix_id_list}, remove_columns=encoded_dataset['train'].column_names, batched=True, load_from_cache_file = False)
        encoded_dataset['test'] = encoded_dataset['test'].map(entity_char_attack, fn_kwargs={'label_to_id': label_to_id, 'tokenizer': tokenizer,'suffix_id_list':suffix_id_list}, remove_columns=encoded_dataset['test'].column_names, batched=True, load_from_cache_file =False)
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
    return logit.argmax(dim=2)[0]

def get_index_select(label_mask):
    index_select = []
    for index in range(len(label_mask)):
        if label_mask[index] == 0:
            index_select.append(index) 
    return index_select

def get_attack_entity_name(entity_info_list, input_ids, tokenizer):
    entitys=[]
    for entity_info in entity_info_list:
        begin,end=entity_info['entity_begin_end']
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
def get_changable_index(begin_end_list):
    res=[]
    for begin,end in begin_end_list:
        num_ids=end-begin
        if num_ids==2:
            res.extend(range(begin+1,begin+2))
        else:
            res.extend(range(begin+1,end-1))
    assert len(res)!=0
    return res
def mydecode_bert(tokenizer,ids):
    res=""
    if hasattr(ids,'tolist'):
        ids=ids.tolist()
    for id in ids:
        s=tokenizer.convert_ids_to_tokens(id)
        if s.startswith("##"):
            res+=s[2:]
        else:
            if res:
                res+=" "+s
            else:
                res+=s
    return res
def mydecode_roberta(tokenizer,ids):
    res=""
    if hasattr(ids,'tolist'):
        ids=ids.tolist()
    for id in ids:
        s=tokenizer.convert_ids_to_tokens(id)
        if s[0]=='Ġ':
            s=' '+s[1:]
        elif s=='</s>':
            s=' '+s
        res+=s
    return res
def mydecode_albert(tokenizer,ids):
    res=""
    if hasattr(ids,'tolist'):
        ids=ids.tolist()
    for id in ids:
        s=tokenizer.convert_ids_to_tokens(id)
        if s[0]=='▁':
            s=' '+s[1:]
        elif s=='[SEP]':
            s=' '+s
        res+=s
    return res
def mydecode(tokenizer,ids):
    res=""
    if tokenizer.name_or_path=='bert-base-uncased':
        res=mydecode_bert(tokenizer,ids)
    elif tokenizer.name_or_path=='gpt2-medium':
        res=mydecode_roberta(tokenizer,ids)
    elif tokenizer.name_or_path=='roberta-base':
        res=mydecode_roberta(tokenizer,ids)
    elif tokenizer.name_or_path=='albert-base-v1':
        res=mydecode_albert(tokenizer,ids)
    elif tokenizer.name_or_path=='xlnet-base-cased':
        res=mydecode_albert(tokenizer,ids)
    else:
        raise Exception("model name problem")
    return res
def attack_success(tokenizer,entity_info,origin_adv_ids,adv_preds,label,pre_attack_word_begin_end_list):
    mismatch_offset=0
    adv_preds=adv_preds.tolist()
    for begin,end in pre_attack_word_begin_end_list:
        new_word=mydecode(tokenizer,origin_adv_ids[range(begin,end)])
        # new_word_ids=tokenizer.encode(new_word,add_special_tokens=False)
        new_sub_word=tokenizer.tokenize(new_word)
        mismatch_offset+=len(new_sub_word)-(end-begin)
    origin_entity_begin=entity_info['entity_begin_end'][0]
    adv_entity_begin=origin_entity_begin+mismatch_offset
    return label[origin_entity_begin] !=adv_preds[adv_entity_begin]

def generate_adversial_text(embeddings, input_ids,entity_info_list, label,log_coeffs,tokenizer, model,args, suffix_id_list=None,suffix_visual_tensor=None,offset=1):
    adv_text, success_attack = "", False
    mismatch=0
    edit_distance=0
    attack_num=0
    # use split to ignore white space in the begining
    origin_tokens=mydecode(tokenizer,get_pure_input_ids(tokenizer,input_ids)).split()
    # for check attack success, construct a attack word list before for each entity
    all_attack_word_begin_end_list=[]
    for entity_info in entity_info_list:
        attack_word_bein_end_list=entity_info['attack_begin_end_list']
        # convert it to tuple
        attack_word_bein_end_list=map(lambda x: (x[0],x[1]),attack_word_bein_end_list)
        all_attack_word_begin_end_list.extend(attack_word_bein_end_list)
    # must remove duplicate
    all_attack_word_begin_end_list=sorted(list(set(all_attack_word_begin_end_list)))
    attack_words_before_each_entity=[[] for _ in range(len(entity_info_list))]
    for i,entity_info in enumerate(entity_info_list):
        entity_begin_end=entity_info['entity_begin_end']
        for begin_end in all_attack_word_begin_end_list:
            if begin_end[1]<=entity_begin_end[0]:
                attack_words_before_each_entity[i].append(begin_end)
    with torch.no_grad():
        # for attack_num in range(args.gumbel_samples):
        log_coeffs=log_coeffs.cpu()
        adv_ids = log_coeffs.argmax(1)
        # should only sample from suffix 4842
        suffix_id_list=torch.LongTensor(suffix_id_list)
        searched_table=dict()
        # first attack, simultaneously evaluate
        for i,entity_info in enumerate(entity_info_list):
            # sample from changable place
            changabel_index=get_changable_index(entity_info['attack_begin_end_list'])
            suffix_candidate_weights=log_coeffs[changabel_index][:,suffix_id_list]
            sub_adv_ids = F.gumbel_softmax(suffix_candidate_weights, hard=True, tau=args.tau).argmax(1)
            candidate_subword_ids=suffix_id_list[sub_adv_ids]
            searched_table[i]=set()
            searched_table[i].add(tuple(candidate_subword_ids.tolist()))
            adv_ids[changabel_index] = candidate_subword_ids
            attack_num+=1
        # predict
        adv_text = mydecode(tokenizer,get_pure_input_ids(tokenizer,adv_ids))
        adv_preds = get_pred(adv_text, tokenizer, model)
        entity_success=0
        # after,check each and attack seperatly until max samples
        for i,entity_info in enumerate(entity_info_list):
            new_entity_ids=[]
            # success
            if attack_success(tokenizer,entity_info,adv_ids,adv_preds,label,attack_words_before_each_entity[i]):
                entity_success+=1
                # edit dis
                adv_tokens=adv_text.split()
                origin_attack_word_list=[origin_tokens[word_index] for word_index in entity_info['attack_word_ids']]
                new_attack_word_list=[adv_tokens[word_index] for word_index in entity_info['attack_word_ids']]
                for origin_word,new_word in zip(origin_attack_word_list,new_attack_word_list):
                    edit_distance+=calculate_edit_distance(origin_word,new_word)

                # mismatch: re encode the new entity string but its ids different from desire entity ids
                for i,(begin,end) in enumerate(entity_info['attack_begin_end_list']):
                    new_word=new_attack_word_list[i]
                    new_word_ids=tokenizer.encode(new_word,add_special_tokens=False)
                    origin_word_ids=adv_ids[range(begin,end)].tolist()
                    if new_word_ids!=origin_word_ids:
                        mismatch+=1
                        break
            # fail
            else:
                changabel_index=get_changable_index(entity_info['attack_begin_end_list'])
                suffix_candidate_weights=log_coeffs[changabel_index][:,suffix_id_list]
                for _ in range(args.gumbel_samples-1):
                    # get new ids
                    for __ in range(10000):
                        sub_adv_ids = F.gumbel_softmax(suffix_candidate_weights, hard=True, tau=args.tau).argmax(1)
                        candidate_subword_ids=suffix_id_list[sub_adv_ids]
                        if tuple(candidate_subword_ids.tolist()) not in searched_table[i]:
                            searched_table[i].add(tuple(candidate_subword_ids.tolist()))
                            break
                    adv_ids[changabel_index] = candidate_subword_ids

                    attack_num+=1

                    # evaluate
                    adv_text = mydecode(tokenizer,get_pure_input_ids(tokenizer,adv_ids))
                    adv_preds = get_pred(adv_text, tokenizer, model)
                    # check success, and break
                    if attack_success(tokenizer,entity_info,adv_ids,adv_preds,label,attack_words_before_each_entity[i]):
                        entity_success+=1
                        # edit dis
                        adv_tokens=adv_text.split()
                        origin_attack_word_list=[origin_tokens[word_index] for word_index in entity_info['attack_word_ids']]
                        new_attack_word_list=[adv_tokens[word_index] for word_index in entity_info['attack_word_ids']]
                        for origin_word,new_word in zip(origin_attack_word_list,new_attack_word_list):
                            edit_distance+=calculate_edit_distance(origin_word,new_word)

                        # mismatch: re encode the new entity string but its ids different from desire entity ids
                        for i,(begin,end) in enumerate(entity_info['attack_begin_end_list']):
                            new_word=new_attack_word_list[i]
                            new_entity_ids=tokenizer.encode(new_word,add_special_tokens=False)
                            if new_entity_ids!=adv_ids[range(begin,end)].tolist():
                                mismatch+=1
                                break
                        break
        if entity_success==len(entity_info_list):
            print("Success!")
            success_attack=True
    return adv_text,entity_success,success_attack,mismatch,attack_num,edit_distance

def generate_adversial_text_compare(embeddings, input_ids,entity_info_list, label, clean_text, log_coeffs, available_indices, tokenizer, model,entity_start_ends,args, suffix_id_list=None,suffix_visual_tensor=None,offset=1):
    return None
    adv_text, success_attack = "", False
    mismatch=0
    edit_distance=0
    attack_num=0
    origin_tokens=mydecode(tokenizer,input_ids).split(' ')[offset:-offset]
    with torch.no_grad():
        # for attack_num in range(args.gumbel_samples):
        log_coeffs=log_coeffs.cpu()
        adv_ids = log_coeffs.argmax(1)
        # should only sample from suffix 4842
        suffix_id_list=torch.LongTensor(suffix_id_list)
        searched_table=dict()
        # first attack, simultaneously evaluate
        for i,entity_info in enumerate(entity_info_list):
            # sample from changable place
            changabel_index=get_changable_index(entity_info['attack_begin_end_list'])
            suffix_candidate_weights=log_coeffs[changabel_index][:,suffix_id_list]
            sub_adv_ids = F.gumbel_softmax(suffix_candidate_weights, hard=True, tau=args.tau).argmax(1)
            candidate_subword_ids=suffix_id_list[sub_adv_ids]
            searched_table[i]=set()
            searched_table[i].add(tuple(candidate_subword_ids.tolist()))
            adv_ids[changabel_index] = candidate_subword_ids
            attack_num+=1
        searched_table1=deepcopy(searched_table)
        adv_ids1=deepcopy(adv_ids)
        # predict
        adv_text = mydecode(tokenizer,adv_ids[offset : - offset])
        adv_preds = get_pred(adv_text, tokenizer, model)
        entity_success_hard_has_mismatch=0
        entity_success_soft=0
        entity_success_no_mismatch=0
        # original
        for i,entity_info in enumerate(entity_info_list):
            new_entity_ids=[]
            # success
            if attack_success(tokenizer,entity_info,adv_text,adv_preds,label):
                entity_success_hard_has_mismatch+=1
            # fail
            else:
                changabel_index=get_changable_index(entity_info['attack_begin_end_list'])
                suffix_candidate_weights=log_coeffs[changabel_index][:,suffix_id_list]
                for _ in range(args.gumbel_samples-1):
                    # get new ids
                    for __ in range(10000):
                        sub_adv_ids = F.gumbel_softmax(suffix_candidate_weights, hard=True, tau=args.tau).argmax(1)
                        candidate_subword_ids=suffix_id_list[sub_adv_ids]
                        if tuple(candidate_subword_ids.tolist()) not in searched_table[i]:
                            searched_table[i].add(tuple(candidate_subword_ids.tolist()))
                            break
                    adv_ids[changabel_index] = candidate_subword_ids
                    attack_num+=1

                    # evaluate
                    _adv_ids = adv_ids[offset : - offset]
                    adv_text = mydecode(tokenizer,_adv_ids)
                    adv_preds = get_pred(adv_text, tokenizer, model)
                    # check success, and break
                    if attack_success(tokenizer,entity_info,adv_text,adv_preds,label):
                        entity_success_hard_has_mismatch+=1
                        break
        
        # soft, no need sample.
        for i,entity_info in enumerate(entity_info_list):
            with torch.no_grad():
                for _ in range(args.gumbel_samples-1):
                    eff=log_coeffs.cuda()
                    probs=F.gumbel_softmax(eff, hard=False, tau=args.tau)
                    new_emb=probs@embeddings.unsqueeze(0)
                    model_output = model(inputs_embeds=new_emb)
                    pred = model_output.logits.cpu()
                    res_label=pred.argmax(dim=-1)[0].tolist()
                    entity_id_pos=entity_info['entity_begin_end'][0]
                    if res_label[entity_id_pos]!=label[entity_id_pos]:
                        entity_success_soft+=1
                        break

        # no mismatch
        for i,entity_info in enumerate(entity_info_list):
            with torch.no_grad():
                new_emb=torch.index_select(embeddings,0,adv_ids1.cuda()).unsqueeze(0)
                model_output = model(inputs_embeds=new_emb)
                pred = model_output.logits.cpu()
                res_label=pred.argmax(dim=-1)[0].tolist()
                entity_id_pos=entity_info['entity_begin_end'][0]

                if res_label[entity_id_pos]!=label[entity_id_pos]:
                    entity_success_no_mismatch+=1
                else:
                    for _ in range(args.gumbel_samples-1):
                        # get new ids
                        for __ in range(10000):
                            sub_adv_ids = F.gumbel_softmax(suffix_candidate_weights, hard=True, tau=args.tau).argmax(1)
                            candidate_subword_ids=suffix_id_list[sub_adv_ids]
                            if tuple(candidate_subword_ids.tolist()) not in searched_table[i]:
                                searched_table1[i].add(tuple(candidate_subword_ids.tolist()))
                                break
                        adv_ids1[changabel_index] = candidate_subword_ids
                        new_emb=torch.index_select(embeddings,0,adv_ids1.cuda()).unsqueeze(0)
                        model_output = model(inputs_embeds=new_emb)
                        pred = model_output.logits.cpu()
                        res_label=pred.argmax(dim=-1)[0].tolist()
                        entity_id_pos=entity_info['entity_begin_end'][0]

                        if res_label[entity_id_pos]!=label[entity_id_pos]:
                            entity_success_no_mismatch+=1
                            break
    return adv_text,entity_success_hard_has_mismatch,success_attack,mismatch,attack_num,edit_distance,entity_success_soft,entity_success_no_mismatch
def check_best_visual(label_mask, input_ids, label, clean_text, log_coeffs, available_indices, tokenizer, model,args, suffix_id_list=None,suffix_visual_tensor=None,offset=1):
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
    