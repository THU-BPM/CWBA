# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import OrderedDict
import json
from bert_score.utils import get_idf_dict
import transformers
import argparse
import jiwer
transformers.logging.set_verbosity(transformers.logging.ERROR)
import time
import torch
import torch.nn.functional as F
from src.sentence_dataset import  load_data, tokenize_and_align_labels, adapt_dataset, generate_adversial_text, get_pred, calculate_edit_distance
from src.utils import bool_flag, print_args, setup_seed, output_data, get_suffix_vocab_list
from src.models import get_model, load_ref_model, get_adversial_loss, get_similarity_loss, get_perplexity_loss, model_prepare
import numpy as np
import matplotlib.pyplot as plt
# from view_data import model_inference
from pre_select_token import get_independent_tokens_bert, get_independent_tokens_roberta, get_independent_tokens_albert
import os

# from src.attack.attack_data import NOT_ATTACK_SENTENCES


def wer(x, y):
    x = " ".join(["%d" % i for i in x])
    y = " ".join(["%d" % i for i in y])

    return jiwer.wer(x, y)


def log_perplexity(logits, coeffs):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()

def draw_pic(sentence, arr, out_file, attack_mask):
    assert len(sentence) == len(arr[0])
    fig, ax = plt.subplots()
    im = ax.imshow(arr, cmap = 'Blues')
    ax.set_xticks(np.arange(len(sentence)), labels=sentence)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for index in range(len(ax.get_xticklabels())):
        if attack_mask[index]:
            ax.get_xticklabels()[index].set_color('r')
    for i in range(len(sentence)):
        text = ax.text(i, 0, arr[0, i], ha="center", va="center", color="r")
    ax.set_title("average gradient for each token")
    fig.tight_layout()
    plt.savefig(out_file)

def get_suffix_visual_tensor(suffix_ids, model):
    if model == 'bert-base-uncased':
        suffix_tensor_map=torch.load("model/suffix_visual_tensor3.pth")
    elif model == 'roberta-base':
        suffix_tensor_map=torch.load("model/suffix_visual_tensor_roberta-base.pth")
    elif model == 'albert-base-v1':
        suffix_tensor_map=torch.load("model/suffix_visual_tensor_albert.pth")
    elif model == 'xlnet-base-cased':
        suffix_tensor_map=torch.load("model/suffix_visual_tensor_xlnet.pth")
    elif model == 'gpt2-medium':
        suffix_tensor_map=torch.load("model/suffix_visual_tensor_gpt.pth")
    result=[]
    for id in suffix_ids:
        result.append(suffix_tensor_map[id])
    result=torch.stack(result).cuda()
    return result
def generate_dataset(adv_texts,origin_ner_tags,dataset_name):
    file_name=os.path.join('data', dataset_name, 'adv_texts.json')
    with open(file_name,'w') as f:
        for adv_text,ner_tags in zip(adv_texts,origin_ner_tags):
            example=OrderedDict()
            example['tokens']=adv_text
            example['ner_tags']=ner_tags
            example['sentence']=" ".join(adv_text)
            json_str=json.dumps(example)
            f.write(json_str+'\n')
        # for example in NOT_ATTACK_SENTENCES:
        #     json_str=json.dumps(example)
        #     f.write(json_str+'\n')

def main(args):
    setup_seed(args.run_id)
    
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # load dataset
    datasets, num_labels, id2label = load_data(args.data_folder, args.num_samples, args.start_index, args.model)
    
    # Load tokenizer, model
    tokenizer, model = get_model(args.model, num_labels, args.model_checkpoint_folder)
    suffix_id_list_other,suffix_list_other, suffix_id_list_all, suffix_list_all = get_suffix_vocab_list(tokenizer, args.model) if args.suffix_gumbel else (None,None)
    # suffix_size * 2048
    suffix_visual_tensor=get_suffix_visual_tensor(suffix_id_list_all, args.model) if args.visual_loss else None
    
    text_key = 'text'
    testset_key = 'validation'
    
    ### get id input
    preprocess_function = lambda examples: tokenizer(examples[text_key], max_length=256, truncation=True)
    
    encoded_dataset = datasets.map(preprocess_function, batched=True)
    
    ### process dataset for white box attack
    encoded_dataset = adapt_dataset(args.target_mode, encoded_dataset, tokenizer, suffix_id_list_all, args)
    
    # Train_data:4123,Test_data:456
    # import ipdb; ipdb.set_trace()
    print(f'Test_data:{len(encoded_dataset[testset_key])}')
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().cuda())

    end_index = min(args.num_samples, len(encoded_dataset[testset_key])/args.max_add)
    json_data = []
    total_mismatch, total_attack_num, total_num, total_success_edit_distance,sentence_success=0,0,0,0,0
    adv_texts=[]
    total_success_attack_number = 0
    for o_idx in range(0, args.num_samples):
        _begin_idx, _end_idx = o_idx * args.max_add, (o_idx + 1) * args.max_add
        for idx in range(_begin_idx, _end_idx):

            input_ids = encoded_dataset[testset_key]['new_input_ids'][idx]
            label_list = encoded_dataset[testset_key]['new_label_list'][idx]
            gradient_mask = encoded_dataset[testset_key]['gradient_mask'][idx]
            attack_indexes = encoded_dataset[testset_key]['attack_indexes'][idx]
            all_tokens = encoded_dataset[testset_key]['all_tokens'][idx]
            print("*"*100)
            print(f"Iter:{idx+1}/{end_index-args.start_index}")
            print('LABEL', label_list)
            print('TEXT', tokenizer.decode(input_ids))
            # import ipdb; ipdb.set_trace()

            available_mask = [0 if item == 1 else 1 for item in gradient_mask]
            forbidden = np.array(gradient_mask).astype('bool')
            available = np.array(available_mask).astype('bool')
            available_indices = np.arange(0, len(input_ids))[available]
            forbidden_indices = torch.from_numpy(np.arange(0, len(input_ids))[forbidden]).cuda()

            log_coeffs= model_prepare(input_ids, embeddings,args)

            start = time.time()
            optimizer = torch.optim.Adam([log_coeffs], lr=args.lr)
            
            # train log_coeffs
            early_stop = False
            first_round_loss = 0
            for i in range(args.num_iters):
                optimizer.zero_grad()
                model.zero_grad()
                # calculate loss， currently only use adv_loss
                # sometimes a sentence(atis 167) contains just O label, no entity
                if len(available_indices)==0:
                    continue
                total_loss, adv_loss, adv_loss1, visual_loss, length_loss= get_adversial_loss(input_ids, args.adv_loss, log_coeffs, available_indices, label_list, num_labels, model, embeddings, args, suffix_id_list_other,suffix_list_other, suffix_id_list_all,suffix_list_all,suffix_visual_tensor)
                # if adv_loss1 < args.kappa:
                #     break
                if i==0:
                    first_round_loss = adv_loss1
                if (i+1) % args.print_every == 0:
                    visual_value = visual_loss.item() if args.visual_loss else 0
                    length_value = length_loss.item() if args.length_loss else 0
                    print(
                        'Iteration %d: total_loss = %.4f,  adv_loss = %.4f,   adv_loss = %.4f, visual_loss = %.4f,  length_loss = %.4f,time=%.2f' % (
                            i + 1, total_loss.item(), adv_loss.item(), adv_loss1.item(), visual_value, length_value, 
                            time.time() - start))
                if (i+1) % args.check_every == 0:
                    if adv_loss1 > args.kappa + 5:
                        early_stop = True
                        break
                # if adv_loss1 < args.kappa:
                #     break
                # Gradient step
                total_loss.backward()
                log_coeffs.grad.index_fill_(0, forbidden_indices, 0)

                optimizer.step()
            if early_stop and idx != _end_idx - 1:
                continue

            print('CLEAN TEXT')
            clean_text = tokenizer.decode(input_ids)

            origin_label = get_pred(clean_text, tokenizer, model).item()
            print(clean_text)

            print('ADVERSARIAL TEXT')
            adv_text, cur_entity_success, adv_ids, attack_number, adv_label = generate_adversial_text(input_ids, label_list, clean_text, log_coeffs, available_indices, tokenizer, model, args, embeddings, suffix_id_list_other,suffix_visual_tensor)
            if (not cur_entity_success) and idx != _end_idx - 1:
                continue
            total_num += 1
            if args.model == 'bert-base-uncased':
                attack_independent_tokens, _ = get_independent_tokens_bert(adv_ids.tolist(), tokenizer)
            elif args.model == 'roberta-base' or args.model == 'gpt2-medium':
                attack_independent_tokens, _ = get_independent_tokens_roberta(adv_ids.tolist(), tokenizer)
            else:
                attack_independent_tokens, _ = get_independent_tokens_albert(adv_ids.tolist(), tokenizer)
            token_contrast_data = []
            edit_distance = 0
            for id in attack_indexes:
                edit_distance += calculate_edit_distance(all_tokens[id], attack_independent_tokens[id])
                token_contrast_data.append([all_tokens[id], attack_independent_tokens[id]])

            print(adv_text)
            if cur_entity_success:
                total_success_edit_distance += edit_distance
                sentence_success += 1
                total_success_attack_number += attack_number
                print('success')
            else:
                print('attack fail!!!')
            total_attack_num += attack_number

            json_data.append({
                'clean_input': clean_text,
                'adv_text': adv_text,
                'origin_label': id2label[str(origin_label)],
                'adv_label': id2label[str(adv_label)],
                'gold_label': id2label[str(label_list)],
                'edit distance': edit_distance,
                'attack_number': attack_number,
                'attack word contrast': token_contrast_data,
            })
            break
    
    output_data(json_data, total_num, sentence_success,total_attack_num, total_success_attack_number, total_success_edit_distance,args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="White-box attack.")

    # save and load
    parser.add_argument("--output_name", default="", type=str,
                        help="specify name")
    parser.add_argument("--model_checkpoint_folder", default='logdir/atis-demo1/pytorch_model.bin', type=str,
                        help="folder for loading GPT2 model trained with BERT tokenizer")    
    parser.add_argument("--data_folder", type=str, default="atis",
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
    parser.add_argument("--check_every", default=100, type=int,
                        help="print loss every x iterations")
    parser.add_argument("--batch_size", default=1, type=int,
                    help="batch size for gumbel-softmax samples")
    parser.add_argument("--lr", default=3e-1, type=float,
                    help="learning rate")
    parser.add_argument("--adv_loss", default="direct", type=str,
                        choices=["cw", "ce", "direct"],
                        help="adversarial loss")
    parser.add_argument("--ind_ent", default=True, type=bool_flag,
                        help="whether only to use the adversial loss")

    # cw loss
    parser.add_argument("--kappa", default=10, type=float,
                        help="CW loss margin")
    
    # gumbel softmax
    parser.add_argument("--gumbel_samples", default=100, type=int,
                        help="number of gumbel samples; if 0, use argmax")
    parser.add_argument("--tau", default=0.5, type=float,
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
    
    parser.add_argument("--initial_select_number", default=5, type=int,
                        help="init select number")
    parser.add_argument("--per_add_number", default=2, type=int,
                        help="add select number per turn")
    parser.add_argument("--max_add", default=20, type=int,
                        help="max add number")
    args = parser.parse_args()
    print_args(args)
    start=time.time()
    main(args)
    print(f"Lasted time:{(time.time()-start)/60} min")