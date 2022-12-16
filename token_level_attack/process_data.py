from datasets import load_dataset
from src.models import get_model
import torch
import numpy as np
import json
from tqdm import tqdm
import os
import time
import argparse
from src.utils import get_pure_input_ids
def load_data():
    filename,label_to_id_filename=None,None
    if model_name=='bert-base-uncased':
        filename='test_temp_processed.json'
        label_to_id_filename='label_to_id.json'
    elif model_name=='roberta-base':
        filename='test_temp.json'
        label_to_id_filename='label_to_id_roberta.json'
    elif model_name=='gpt2-medium':
        filename='test_temp.json'
        label_to_id_filename='label_to_id_gpt.json'
    elif model_name=='albert-base-v1':
        filename='test_temp.json'
        label_to_id_filename='label_to_id_albert.json'
    elif model_name=='xlnet-base-cased':
        filename='test_temp.json'
        label_to_id_filename='label_to_id_xlnet.json'
    else:
        raise Exception('model name error')
    
    data_files = {
        split_key: os.path.join('data', data_folder, filename)
    }
    datasets = load_dataset('json', data_files=data_files)

    with open(os.path.join('data', data_folder, label_to_id_filename),'r') as f:
        label_to_id = json.load(f)
    return datasets,label_to_id

def get_independent_words(word):
    '''eg: 'a&b'->'a' '&' 'b',to keep encoding and decoding consistant'''
    ids=tokenizer.encode(word,add_special_tokens=False)
    word_list = []
    parts_of_previous_word = []
    for id in ids:
        token = tokenizer.convert_ids_to_tokens(id)
        if len(parts_of_previous_word) == 0:
            parts_of_previous_word.append(id)
        elif token.startswith('##'):
            parts_of_previous_word.append(id)
        else:
            word_list.append(parts_of_previous_word)
            parts_of_previous_word = [id]
    word_list.append(parts_of_previous_word)
    result_word_list = []
    for subwords in word_list:
        result_word_list.append(tokenizer.decode(subwords))
    return result_word_list
def get_append_label(current_label):
    if current_label == 'O':
        return current_label
    else:
        return 'I-'+ current_label[2:]
def get_independent_tokens_and_new_ner_tags(tokens,ner_tags):
    assert len(tokens)==len(ner_tags)
    res_tokens=[]
    new_ner_tags=[]
    for i,word in enumerate(tokens):
        independent_words=get_independent_words(word)
        cur_new_tags=[ner_tags[i]]
        append_label=get_append_label(ner_tags[i])
        for _ in range(len(independent_words)-1): cur_new_tags.append(append_label)
        res_tokens.extend(independent_words)
        new_ner_tags.extend(cur_new_tags)
    return res_tokens,new_ner_tags

def get_word_boundaries_in_input_ids(independent_tokens):
    word_boundaries=[]
    begin=0
    for token in independent_tokens:
        sub_tokens=tokenizer.tokenize(token)
        word_boundaries.append((begin,begin+len(sub_tokens)))
        begin+=len(sub_tokens)
    return word_boundaries
def get_entity_word_indices(new_ner_tags):
    res=[]
    for i,tag in enumerate(new_ner_tags):
        if tag.startswith('B-'):
            res.append(i)
    return res
def get_word_rank_by_gradient(tokens, ner_tags):
    entity_info_list=[]
    sentence=" ".join(tokens)
    word_boundaries_in_input_ids=get_word_boundaries_in_input_ids(tokens)
    entity_word_indices=get_entity_word_indices(ner_tags)
    input_ids = tokenizer(sentence, return_tensors="pt")['input_ids'][0].cuda()
    with torch.no_grad():
        input_embeddings=torch.index_select(embeddings,0,input_ids).unsqueeze(0)
        input_embeddings.requires_grad = True
    for entity_index_in_words in entity_word_indices:
        entity_info=dict()
        model.zero_grad()
        if input_embeddings.grad is not None:
            input_embeddings.grad.zero_()
        model_output = model(inputs_embeds=input_embeddings)
        # (1*T*L)
        pred = model_output.logits
        offset=0 if 'gpt' or 'xlnet' in model_name else 1
        entity_index_in_ids=word_boundaries_in_input_ids[entity_index_in_words][0]+offset
        entity_logits=pred[0,entity_index_in_ids]
        target_label=ner_tags[entity_index_in_words]
        target_label=label_to_id[target_label]
        top_preds=entity_logits.sort(descending=True)[1]
        adv_logit=top_preds[0]
        if top_preds[0]==target_label:
            adv_logit=top_preds[1]
        loss=(entity_logits[target_label]-entity_logits[adv_logit]+kappa).clamp(min=0)
        loss.backward()
        gradient = torch.norm(input_embeddings.grad, dim=2)[0].detach().tolist()
        gradient = get_pure_input_ids(tokenizer,gradient)
        agg_gradient=[]
        for begin_end in word_boundaries_in_input_ids:
            agg_gradient.append(np.mean(gradient[begin_end[0]:begin_end[1]]))
        ranked = np.argsort(agg_gradient)
        largest_indices = ranked[::-1][:len(ranked)].tolist()

        entity_info['index_in_words']=entity_index_in_words
        entity_info['gradient_rank']=largest_indices
        entity_info_list.append(entity_info)
    return entity_info_list

def process_data(input_data, output_file):
    f = open(output_file, 'w')
    for index in tqdm(range(len(datasets[split_key]))):
        tokens = input_data[split_key][index]['tokens']
        ner_tags = input_data[split_key][index]['ner_tags']
        # independent_tokens,new_ner_tags=get_independent_tokens_and_new_ner_tags(tokens,ner_tags)
        entity_info_list = get_word_rank_by_gradient(tokens, ner_tags)
        item = {
            'tokens': tokens,
            'ner_tags': ner_tags,
            'sentence':" ".join(tokens),
            'entity_info_list': entity_info_list,
        }
        f.write(json.dumps(item) + '\n')
    f.close()

def parse():
    parser = argparse.ArgumentParser(description="White-box attack.")
    # parser.add_argument("--data_filename", type=str, default="test_temp_processed.json",
    #                     help="the filename storing data")
    parser.add_argument("--model_checkpoint_folder", default='logdir/atis-demo1/pytorch_model.bin', type=str,
                        help="folder for loading GPT2 model trained with BERT tokenizer")    
    parser.add_argument("--data_folder", type=str, default="atis",
                        help="folder in which to store data")
    parser.add_argument("--label_to_id_filename", type=str, default="label_to_id.json",
                        help="the label to id filename for different model"),
    parser.add_argument("--model", default="bert-base-uncased", type=str,
                        help="type of model")
    parser.add_argument('--output_name',type=str,default='test_processed.json',help='processed_dataset_name')
    parser.add_argument("--kappa", default=5, type=float,
                        help="gradient loss margin")
    args = parser.parse_args()
    return args
args=parse()
print(args)
s=time.time()
split_key='test'
data_folder=args.data_folder
model_name=args.model
model_checkpoint_folder=args.model_checkpoint_folder
kappa=args.kappa
datasets,label_to_id = load_data()
tokenizer, model = get_model(model_name, len(label_to_id), model_checkpoint_folder)
embeddings = model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().cuda())
process_data(datasets, f'data/{data_folder}/{args.output_name}')
print(f'Last:{(time.time()-s)/60}min')
