# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import OrderedDict
import json
import transformers
import argparse
transformers.logging.set_verbosity(transformers.logging.ERROR)
import time
import torch
from src.token_dataset import  tokenize_and_align_labels, adapt_dataset, generate_adversial_text, get_attack_entity_name
from src.utils import bool_flag, print_args, setup_seed, output_data, get_suffix_vocab_list
from src.models import get_model, get_adversial_loss, init_log_coeffs
import numpy as np
from view_data import model_inference
import os
from src.attack.attack_data import NOT_ATTACK_SENTENCES
from datasets import load_dataset

def load_data():
    filename,label_to_id_filename=None,None
    model_name=args.model
    if model_name=='bert-base-uncased':
        filename='test_processed.json'
        label_to_id_filename='label_to_id.json'
    elif model_name=='roberta-base':
        filename='test_processed_roberta.json'
        label_to_id_filename='label_to_id_roberta.json'
    elif model_name=='gpt2-medium':
        filename='test_processed_gpt.json'
        label_to_id_filename='label_to_id_gpt.json'
    elif model_name=='albert-base-v1':
        filename='test_processed_albert.json'
        label_to_id_filename='label_to_id_albert.json'
    elif model_name=='xlnet-base-cased':
        filename='test_processed_xlnet.json'
        label_to_id_filename='label_to_id_xlnet.json'
    else:
        raise Exception('model name error')
    data_files = {
        'test': os.path.join('data', args.data_folder, filename),
    }
    datasets = load_dataset('json', data_files=data_files)
    with open(os.path.join('data', args.data_folder, label_to_id_filename),'r') as f:
        label_to_id = json.load(f)
    id_to_label = {id: label for label, id in label_to_id.items()}
    num_labels = len(label_to_id)
    return datasets, label_to_id, id_to_label, num_labels

def get_suffix_visual_tensor(suffix_ids):
    suffix_tensor_map=torch.load(f"suffix_visual_tensor_{args.model}.pth")
    result=[]
    for id in suffix_ids:
        result.append(suffix_tensor_map[id])
    result=torch.stack(result).cuda()
    return result
def generate_dataset(adv_texts,origin_ner_tags,dataset_name):
    file_name=os.path.join('data', dataset_name, f'adv_texts_{args.model}.json')
    with open(file_name,'w') as f:
        for adv_text,ner_tags in zip(adv_texts,origin_ner_tags):
            example=OrderedDict()
            example['tokens']=adv_text
            example['ner_tags']=ner_tags
            example['sentence']=" ".join(adv_text)
            json_str=json.dumps(example,ensure_ascii=False)
            f.write(json_str+'\n')
        for example in NOT_ATTACK_SENTENCES:
            json_str=json.dumps(example,ensure_ascii=False)
            f.write(json_str+'\n')
def is_first_attack(idx):
    return (idx%2)==0

def main(args):
    setup_seed(args.run_id)

    os.environ["CUDA_VISIBLE_DEVICES"]="4"
    # load dataset
    datasets, label_to_id, id_to_label, num_labels = load_data()
    id_to_label[-100] = 'O'

    # Load tokenizer, model
    tokenizer, model = get_model(args.model, num_labels, args.model_checkpoint_folder)
    suffix_id_list,suffix_list = get_suffix_vocab_list(tokenizer,adding_pad=False,model_name=args.model) if args.suffix_gumbel else (None,None)
    # suffix_size * 2048
    suffix_visual_tensor=get_suffix_visual_tensor(suffix_id_list) if args.visual_loss else None
    testset_key = 'test'
    # align labels to match bert input subwords
    datasets=datasets.filter(lambda example,index: index<101,with_indices=True)
    encoded_dataset = datasets.map(tokenize_and_align_labels, batched=True, fn_kwargs = {'tokenizer': tokenizer, 'label_to_id': label_to_id}, load_from_cache_file =False)
    ### process dataset for white box attack
    encoded_dataset = adapt_dataset(args.target_mode, encoded_dataset, label_to_id, tokenizer,suffix_id_list)
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().cuda())

    end_index = min(args.start_index + args.num_samples, len(encoded_dataset[testset_key]))
    json_data=[]
    total_entity,entity_success, total_output,total_mismatch, total_attack_num,total_success_edit_distance,sentence_success=0,0,0,0,0,0,0
    fail_ids=[]
    adv_texts=[]
    origin_ner_tags=[]
    # args.start_index
    idx=args.start_index*2
    while(idx<end_index*2):
        input_ids = encoded_dataset[testset_key]['new_input_ids'][idx]
        token_type_ids = encoded_dataset[testset_key]['token_type_ids'][idx] if args.model != 'gpt2' else None
        label = encoded_dataset[testset_key]['origin_labels'][idx]
        noise_label = encoded_dataset[testset_key]['new_label_list'][idx]
        gradient_mask = encoded_dataset[testset_key]['gradient_mask'][idx]
        label_mask = encoded_dataset[testset_key]['label_mask'][idx]
        entity_start_ends=encoded_dataset[testset_key]['entity_start_ends'][idx]
        entity_info_list=encoded_dataset[testset_key]['entity_info_list'][idx]
        print("*"*100)
        print(f"Iter:{idx//2}/{end_index-args.start_index}:{idx%2}")
        print('LABEL', label)
        print('TEXT', tokenizer.decode(input_ids))

        offset = 0 if 'gpt' in args.model else 1

        available_mask = [0 if item == 1 else 1 for item in gradient_mask]
        forbidden = np.array(gradient_mask).astype('bool')
        available = np.array(available_mask).astype('bool')

        available_indices = np.arange(0, len(input_ids))[available]
        forbidden_indices = torch.from_numpy(np.arange(0, len(input_ids))[forbidden]).cuda()

        token_type_ids_batch = (None if token_type_ids is None else torch.LongTensor(token_type_ids).unsqueeze(0).repeat(args.batch_size,1).cuda())

        log_coeffs= init_log_coeffs(input_ids,embeddings,args)

        start = time.time()
        optimizer = torch.optim.Adam([log_coeffs], lr=args.lr)
    
        # first optimize log_coeffs
        for i in range(args.num_iters):
            optimizer.zero_grad()
            model.zero_grad()
            # sometimes a sentence(atis 167) contains just O label, no entity
            if len(available_indices)==0:
                continue
            total_loss, adv_loss, adv_loss1, visual_loss, length_loss = get_adversial_loss(input_ids,entity_info_list,args.adv_loss, log_coeffs, available_indices, label, noise_label, label_mask, num_labels, model, embeddings,entity_start_ends,token_type_ids_batch, args, suffix_id_list,suffix_list,suffix_visual_tensor)
            if (i+1) % args.print_every == 0:
                    visual_value = visual_loss.item() if args.visual_loss else 0
                    length_value = length_loss.item() if args.length_loss else 0
                    print(
                        'Iteration %d: total_loss = %.4f,  adv_loss_soft = %.4f,   adv_loss_hard = %.4f, visual_loss = %.4f,  length_loss = %.4f,time=%.2f' % (
                            i + 1, total_loss.item(), adv_loss.item(), adv_loss1.item(), visual_value, length_value, 
                            time.time() - start))

            # Gradient step
            total_loss.backward()
            log_coeffs.grad.index_fill_(0, forbidden_indices, 0)
            optimizer.step()
        # import ipdb; ipdb.set_trace()
        print('CLEAN TEXT')
        clean_text = tokenizer.decode(input_ids[offset:(len(input_ids) - offset)])
        print(clean_text)
        print('target entity')
        target_entity = get_attack_entity_name(entity_info_list, input_ids, tokenizer)
        print(target_entity)
    
        # then sample adversarial text
        adv_text,cur_entity_success,success_attack,mismatch,attack_num,edit_distance = generate_adversial_text(embeddings,input_ids, entity_info_list, label, log_coeffs,tokenizer, model,args,suffix_id_list,suffix_visual_tensor,offset=offset)
        print('ADVERSARIAL TEXT')
        print(adv_text)

        total_attack_num += attack_num
        origin_idx=idx
        if is_first_attack(idx) and success_attack:
            idx+=2
        # continue means: don't collect statistics
        elif is_first_attack(idx) and not success_attack:
            idx+=1
            continue
        # second attack both sucess or fail will collect statistics
        else:
            idx+=1
        ner_tags = encoded_dataset[testset_key]['ner_tags'][origin_idx]
        if len(ner_tags)!=len(adv_text.split()):
            print("!!!!NER_ADV_NOT_MATCH!!!!!")
            print(len(ner_tags))
            print(len(adv_text.split()))
            print(ner_tags)
            print(adv_text.split())
            origin_ner_tags.insert(0,ner_tags)
            adv_texts.insert(0,adv_text.split())
        origin_ner_tags.append(ner_tags)
        adv_texts.append(adv_text.split())

        # finally statistics
        total_entity+=len(entity_info_list)
        if success_attack:
            sentence_success+=1
        else:
            print(f"FAILED!:{origin_idx//2}")
            print(f"success:{cur_entity_success}/total:{len(entity_info_list)}")
            fail_ids.append(origin_idx//2)
        total_success_edit_distance+=edit_distance
        total_mismatch += mismatch
        entity_success += cur_entity_success
        adv_pred_label_str, adv_str_list_ann =  model_inference(model, tokenizer, adv_text, id_to_label)
        if len(adv_str_list_ann) == len(label):
            adv_str_list_ann_contrastive = [adv_str_list_ann[i] + ' (' + id_to_label[label[i]] + ')' for i in range(len(label))]
            json_data.append({
                'target_entity': target_entity,
                'edit_distance': edit_distance,
                'clean_text': clean_text,
                'adv_text': adv_text,
                'adv_str_list_ann_contrastive': adv_str_list_ann_contrastive,
                'success': success_attack
            })
            total_output += 1
    if args.generate_dataset:
        generate_dataset(adv_texts,origin_ner_tags,args.data_folder)
    output_data(json_data,end_index - args.start_index,total_entity, entity_success,sentence_success,total_mismatch, total_output,total_attack_num, fail_ids,total_success_edit_distance,args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="White-box attack.")

    # save and load
    parser.add_argument("--output_name", default="", type=str,
                        help="specify name")
    parser.add_argument("--model_checkpoint_folder", default='logdir/atis-demo1/pytorch_model.bin', type=str,
                        help="folder for loading GPT2 model trained with BERT tokenizer")    
    parser.add_argument("--data_folder", type=str, default="atis",
                        help="folder in which to store data")
    parser.add_argument("--data_filename", type=str, default="test_processed.json",
                        help="the filename storing data")
    parser.add_argument("--label_to_id_filename", type=str, default="label_to_id.json",
                        help="folder in which to store data")
    parser.add_argument("--model", default="bert-base-uncased", type=str,
                        help="type of model")
    parser.add_argument("--run_id", default=0, type=int,
                        help="run_id_for_save")
    parser.add_argument("--start_index", default=0, type=int,
                        help="starting sample index")
    parser.add_argument("--generate_dataset", default=False, type=bool_flag,
                        help="if generate a adv dataset in data folder")

    # optimize
    parser.add_argument("--num_samples", default=1, type=int,
                        help="number of samples to attack")
    parser.add_argument("--num_iters", default=10, type=int,
                    help="number of epochs to train for")
    parser.add_argument("--print_every", default=2, type=int,
                        help="print loss every x iterations")
    parser.add_argument("--batch_size", default=10, type=int,
                    help="batch size for gumbel-softmax samples")
    parser.add_argument("--lr", default=3e-1, type=float,
                    help="learning rate")
    parser.add_argument("--adv_loss", default="direct", type=str,
                        choices=["cw", "ce", "direct"],
                        help="adversarial loss")
    parser.add_argument("--ind_ent", default=True, type=bool_flag,
                        help="whether only to use the adversial loss")

    # cw loss
    parser.add_argument("--kappa", default=5, type=float,
                        help="CW loss margin")
    
    # gumbel softmax
    parser.add_argument("--gumbel_samples", default=100, type=int,
                        help="number of gumbel samples; if 0, use argmax")
    parser.add_argument("--tau", default=1, type=float,
                        help="tau for gumbel softmax")
    parser.add_argument("--initial_coeff", default=1, type=int,
                        help="initial log coefficients")
    parser.add_argument("--suffix_gumbel", default=False, type=bool_flag,
                        help="initial log coefficients")
    
    # mode setting
    parser.add_argument("--target_mode", default="insert_1", type=str,
                        help="target attack mode example insert_l_r, nearby_l_r, char_entity")

    # Attack setting
    parser.add_argument("--constraint", default="bertscore_idf", type=str,
                        choices=["cosine", "bertscore", "bertscore_idf"],
                        help="constraint function")
    parser.add_argument("--embed_layer", default=-1, type=int,
                        help="which layer of LM to extract embeddings from")
    parser.add_argument("--lam_sim", default=1, type=float,
                        help="embedding similarity regularizer")
    parser.add_argument("--lam_perp", default=1, type=float,
                        help="(log) perplexity regularizer")
    parser.add_argument("--only_adv", default=True, type=bool_flag,
                        help="whether only to use the adversial loss")
    parser.add_argument("--visual_loss",default=False, type=bool_flag,
                        help="whether use word image similarity constraint")
    parser.add_argument("--length_loss",default=False, type=bool_flag,
                        help="whether use word length difference constraint")
    parser.add_argument("--lam_visual", default=1, type=float,
                        help="visual embedding similarity regularizer")
    parser.add_argument("--lam_length", default=1, type=float,
                        help="length difference regularizer")
    args = parser.parse_args()
    print_args(args)
    start=time.time()
    main(args)
    print(f"Lasted time:{(time.time()-start)/60} min")