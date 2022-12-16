# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import enum
from tqdm import tqdm
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Config
import numpy as np
import random
import json
import sys


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}
def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

# offset target by 1 if labels start from 1
def target_offset(examples):
    examples["label"] = list(map(lambda x: x - 1, examples["label"]))
    return examples

def get_output_file(args, model, start, end):
    suffix = ''
    if args.finetune:
        suffix += '_finetune'
    if args.dataset == 'mnli':
        dataset_str = f"{args.dataset}_{args.mnli_option}_{args.attack_target}"
    else:
        dataset_str = args.dataset
    attack_str = args.adv_loss
    if args.adv_loss == 'cw':
        attack_str += f'_kappa={args.kappa}'

    model_name = model.replace('/', '-')
    output_file = f"{model_name}_{dataset_str}{suffix}_{start}-{end}"
    output_file += f"_iters={args.num_iters}_{attack_str}_lambda_sim={args.lam_sim}_lambda_perp={args.lam_perp}_emblayer={args.embed_layer}_{args.constraint}.pth"

    return output_file

def load_checkpoints(args):
    if args.dataset == 'mnli':
        adv_log_coeffs = {'premise': [], 'hypothesis': []}
        clean_texts = {'premise': [], 'hypothesis': []}
        adv_texts = {'premise': [], 'hypothesis': []}
    else:
        adv_log_coeffs, clean_texts, adv_texts = [], [], []
    clean_logits, adv_logits, times, labels = [], [], [], []
    
    for i in tqdm(range(args.start_index, args.end_index, args.num_samples)):
        output_file = get_output_file(args, args.surrogate_model, i, i + args.num_samples)
        output_file = os.path.join(args.adv_samples_folder, output_file)
        if os.path.exists(output_file):
            checkpoint = torch.load(output_file)
            clean_logits.append(checkpoint['clean_logits'])
            adv_logits.append(checkpoint['adv_logits'])
            labels += checkpoint['labels']
            times += checkpoint['times']
            if args.dataset == 'mnli':
                adv_log_coeffs['premise'] += checkpoint['adv_log_coeffs']['premise']
                adv_log_coeffs['hypothesis'] += checkpoint['adv_log_coeffs']['hypothesis']
                clean_texts['premise'] += checkpoint['clean_texts']['premise']
                clean_texts['hypothesis'] += checkpoint['clean_texts']['hypothesis']
                adv_texts['premise'] += checkpoint['adv_texts']['premise']
                adv_texts['hypothesis'] += checkpoint['adv_texts']['hypothesis']
            else:
                adv_log_coeffs += checkpoint['adv_log_coeffs']
                clean_texts += checkpoint['clean_texts']
                adv_texts += checkpoint['adv_texts']
        else:
            print('Skipping %s' % output_file)
    clean_logits = torch.cat(clean_logits, 0)
    adv_logits = torch.cat(adv_logits, 0)
    return clean_texts, adv_texts, clean_logits, adv_logits, adv_log_coeffs, labels, times

def print_args(args):
    args_dict = vars(args)
    for arg_name, arg_value in sorted(args_dict.items()):
        print(f"\t{arg_name}: {arg_value}")
        
def embedding_from_weights(w):
    layer = torch.nn.Embedding(w.size(0), w.size(1))
    layer.weight.data = w

    return layer

def load_gpt2_from_dict(dict_path, output_hidden_states=False):
    state_dict = torch.load(dict_path)['model']

    config = GPT2Config(
        vocab_size=30522,
        n_embd=1024,
        n_head=8,
        activation_function='relu',
        n_layer=24,
        output_hidden_states=output_hidden_states
    )
    model = GPT2LMHeadModel(config)
    model.load_state_dict(state_dict)
    # The input embedding is not loaded automatically
    model.set_input_embeddings(embedding_from_weights(state_dict['transformer.wte.weight'].cpu()))

    return model

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def output_data(json_data,total_sentence,total_entity,entity_success,sentence_success,total_mismatch, total_output,total_attack_num,fail_ids,total_success_edit_distance,args):
    cfg = vars(args)
    meta_result = {
        'result': {
            'total_sentence': total_sentence,
            'total_entity': total_entity,
            'total_output': total_output, 
            'entity_success': entity_success,
            'sentence_success':sentence_success,
            'total_mismatch':total_mismatch,
            'success_rate': entity_success/total_entity,
            'mismatch_rate':total_mismatch/entity_success,
            "total_attack_num":total_attack_num,
            "average_attack_num":total_attack_num/total_entity,
            "success_attack_num":(total_attack_num-(total_entity-entity_success)*args.gumbel_samples*2),
            "average_success_attack_num":(total_attack_num-(total_entity-entity_success)*args.gumbel_samples*2)/entity_success,
            "average_success_edit_distance": total_success_edit_distance/entity_success,
            "fail_ids":fail_ids,
        },
        'config': cfg
    }
    output_file = (f"new_sample/{args.model}/"
                f"dataset_{args.data_folder}_number_{args.num_samples}_run_id_{args.run_id}_name_{args.output_name}")
    os.makedirs(output_file, exist_ok=True)
    meta_output = os.path.join(output_file, 'meta.json')
    sample_output = os.path.join(output_file, 'sample.json')
    command_output = os.path.join(output_file, 'command.sh')
    
    with open(meta_output, 'w') as f:
        json.dump(meta_result, f, indent=4)
    
    with open(sample_output, 'w') as f:
        json.dump(json_data, f, indent=4,ensure_ascii=False)
    
    with open(command_output, 'w') as f:
        f.write("CUDA_VISIBLE_DEVICES=1 python "+" ".join(sys.argv))

def get_suffix_vocab_list(tokenizer, adding_pad=False,model_name='bert-base-uncased'):
    words_info = [(tok, ids) for tok, ids in tokenizer.vocab.items()]
    words_info=sorted(words_info)
    suffix_id_list = []
    suffix_list=[]
    if adding_pad:
        suffix_id_list.append(tokenizer.encode('[PAD]')[1])
    if model_name=='bert-base-uncased':
        for word, id in words_info:
            if word.startswith('##'):
                suffix_id_list.append(id)
                suffix_list.append(word)
    elif model_name=='roberta-base' or model_name=='gpt2-medium':
        for word, id in words_info:
            if not word.startswith('Ġ'):
                suffix_id_list.append(id)
                suffix_list.append(word)
    elif model_name=='xlnet-base-cased' or model_name=='albert-base-v1':
        for word, id in words_info:
            if not word.startswith('▁'):
                suffix_id_list.append(id)
                suffix_list.append(word)
    else:
        raise Exception('model name error')
    return suffix_id_list,suffix_list
def bert_ids_to_word_begin_end_list(tokenizer,input_ids):
    def is_suffix(id):
        return tokenizer.convert_ids_to_tokens(id).startswith('##')
    input_ids=input_ids[1:-1]
    begin_end_list=[]
    n=len(input_ids)
    begin=0
    while(begin<n):
        end=begin+1
        while(end<n and is_suffix(input_ids[end])): end+=1
        # +1 means consider [CLS]
        begin_end_list.append((begin+1,end+1))
        begin=end
    return begin_end_list
def gpt_ids_to_word_begin_end_list(tokenizer,input_ids):
    def is_suffix(id):
        return not tokenizer.convert_ids_to_tokens(id).startswith('Ġ')
    begin_end_list=[]
    n=len(input_ids)
    begin=0
    while(begin<n):
        end=begin+1
        while(end<n and is_suffix(input_ids[end])): end+=1
        begin_end_list.append((begin,end))
        begin=end
    return begin_end_list

def roberta_ids_to_word_begin_end_list(tokenizer,input_ids):
    def is_suffix(id):
        return not tokenizer.convert_ids_to_tokens(id).startswith('Ġ')
    input_ids=input_ids[1:-1]
    begin_end_list=[]
    n=len(input_ids)
    begin=0
    while(begin<n):
        end=begin+1
        while(end<n and is_suffix(input_ids[end])): end+=1
        begin_end_list.append((begin+1,end+1))
        begin=end
    return begin_end_list
def albert_ids_to_word_begin_end_list(tokenizer,input_ids):
    def is_suffix(id):
        return not tokenizer.convert_ids_to_tokens(id).startswith('▁')
    input_ids=input_ids[1:-1]
    begin_end_list=[]
    n=len(input_ids)
    begin=0
    while(begin<n):
        end=begin+1
        while(end<n and is_suffix(input_ids[end])): end+=1
        begin_end_list.append((begin+1,end+1))
        begin=end
    return begin_end_list
def xlnet_ids_to_word_begin_end_list(tokenizer,input_ids):
    def is_suffix(id):
        return not tokenizer.convert_ids_to_tokens(id).startswith('▁')
    input_ids=input_ids[:-2]
    begin_end_list=[]
    n=len(input_ids)
    begin=0
    while(begin<n):
        end=begin+1
        while(end<n and is_suffix(input_ids[end])): end+=1
        begin_end_list.append((begin,end))
        begin=end
    return begin_end_list
def ids_to_word_begin_end_list(tokenizer,input_ids):
    '''bert input_ids to word begin end list,offset=1 means bert ([CLS] and [SEP]),0 means gpt
    return begin end of each word in ids
    '''
    if hasattr(input_ids,'tolist'):
        input_ids=input_ids.tolist()
    if tokenizer.name_or_path=='bert-base-uncased':
        begin_end_list=bert_ids_to_word_begin_end_list(tokenizer,input_ids)
    elif tokenizer.name_or_path=='gpt2-medium':
        begin_end_list=gpt_ids_to_word_begin_end_list(tokenizer,input_ids)
    elif tokenizer.name_or_path=='roberta-base':
        begin_end_list=roberta_ids_to_word_begin_end_list(tokenizer,input_ids)
    elif tokenizer.name_or_path=='albert-base-v1':
        begin_end_list=albert_ids_to_word_begin_end_list(tokenizer,input_ids)
    elif tokenizer.name_or_path=='xlnet-base-cased':
        begin_end_list=xlnet_ids_to_word_begin_end_list(tokenizer,input_ids)
    else:
        raise Exception("model name problem")
    return begin_end_list

def get_pure_input_ids(tokenizer,input_ids):
    model_name=tokenizer.name_or_path
    if model_name=='bert-base-uncased':
        return input_ids[1:-1]
    elif model_name=='roberta-base':
        return input_ids[1:-1]
    elif model_name=='gpt2-medium':
        return input_ids[:]
    elif model_name=='albert-base-v1':
        return input_ids[1:-1]
    elif model_name=='xlnet-base-cased':
        return input_ids[:-2]
    else:
        raise Exception('model name error')
    
