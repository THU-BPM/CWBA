from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from src.sentence_dataset import load_data
import torch
import numpy as np
import json
from tqdm import tqdm
import argparse
import os
from datasets import load_dataset


def load_data(dataset_name, file_suffix):
    data_files = {
        'train': os.path.join('data', dataset_name, 'train_' + file_suffix +'.json'),
        'validation': os.path.join('data', dataset_name, 'test_' + file_suffix + '.json')
    }
    with open(os.path.join('data', dataset_name, 'id2label.json'), 'r') as f:
        id2label = json.load(f)
    datasets = load_dataset('json', data_files=data_files)
    
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


def get_independent_tokens_bert(ids, tokenizer):
    '''eg: 'a&b'->'a' '&' 'b' '''
    token_list = []
    current_token_list = []
    begin_end_list = []
    begin = 0
    for index in range(len(ids)):
        id = ids[index]
        token = tokenizer.convert_ids_to_tokens(id)
        if len(current_token_list) == 0:
            current_token_list.append(id)
        elif token.startswith('##'):
            current_token_list.append(id)
        else:
            token_list.append(current_token_list)
            begin_end_list.append([begin, begin + len(current_token_list)])
            begin += len(current_token_list)
            current_token_list = [id]
    token_list.append(current_token_list)
    begin_end_list.append([begin, begin + len(current_token_list)])
    result_token_list = []
    for item in token_list:
        result_token_list.append(tokenizer.decode(item))
    return result_token_list, begin_end_list


def get_independent_tokens_roberta(ids, tokenizer):
    '''eg: 'a&b'->'a' '&' 'b' '''
    token_list = []
    current_token_list = []
    begin_end_list = []
    begin = 0
    for index in range(len(ids)):
        id = ids[index]
        token = tokenizer.convert_ids_to_tokens(id)
        if len(current_token_list) == 0:
            current_token_list.append(id)
        elif not token.startswith('Ġ'):
            current_token_list.append(id)
        else:
            token_list.append(current_token_list)
            begin_end_list.append([begin, begin + len(current_token_list)])
            begin += len(current_token_list)
            current_token_list = [id]
    token_list.append(current_token_list)
    begin_end_list.append([begin, begin + len(current_token_list)])
    result_token_list = []
    for item in token_list:
        result_token_list.append(tokenizer.decode(item))
    return result_token_list, begin_end_list

def get_independent_tokens_albert(ids, tokenizer):
    '''eg: 'a&b'->'a' '&' 'b' '''
    token_list = []
    current_token_list = []
    begin_end_list = []
    begin = 0
    for index in range(len(ids)):
        id = ids[index]
        token = tokenizer.convert_ids_to_tokens(id)
        if len(current_token_list) == 0:
            current_token_list.append(id)
        elif not token.startswith('▁'):
            current_token_list.append(id)
        else:
            token_list.append(current_token_list)
            begin_end_list.append([begin, begin + len(current_token_list)])
            begin += len(current_token_list)
            current_token_list = [id]
    token_list.append(current_token_list)
    begin_end_list.append([begin, begin + len(current_token_list)])
    result_token_list = []
    for item in token_list:
        result_token_list.append(tokenizer.decode(item))
    return result_token_list, begin_end_list

def get_independent_token_and_rank(text, label, tokenizer, model, embeddings, model_name):
    model.zero_grad()
    inputs = tokenizer(text, return_tensors="pt")
    # begin_end is independent word's [begin end) idx in input_ids
    if model_name == 'bert-base-uncased':
        result_token_list, begin_end_list = get_independent_tokens_bert(inputs['input_ids'][0][1:-1].tolist(), tokenizer)
    elif model_name == 'roberta-base' or model_name == 'gpt2-medium':
        result_token_list, begin_end_list = get_independent_tokens_roberta(inputs['input_ids'][0][1:-1].tolist(), tokenizer)
    else:
        result_token_list, begin_end_list = get_independent_tokens_albert(inputs['input_ids'][0][1:-1].tolist(), tokenizer)
    with torch.no_grad():
        # log_coeffs = torch.zeros(len(inputs['input_ids'][0]), embeddings.size(0))
        # indices = torch.arange(log_coeffs.size(0)).long()
        # log_coeffs[indices, inputs['input_ids'][0]] = 1
        # input_embeddings = log_coeffs@embeddings[None, :, :]
        input_embeddings = embeddings[None, :, :][:, inputs['input_ids'][0]]
        input_embeddings = input_embeddings.cuda()
        input_embeddings.requires_grad = True
        # import ipdb; ipdb.set_trace()
    
    model_output = model(inputs_embeds=input_embeddings)
    pred = model_output.logits
    top_preds = pred.sort(descending=True)[1]
    correct = (top_preds[:, 0] == label).long()
    indices = top_preds.gather(1, correct.view(-1, 1))
    adv_loss = (pred[:, label] - pred.gather(1, indices) + 5).clamp(min=0).mean()
    adv_loss.backward()
    gradient = torch.norm(input_embeddings.grad, dim=2)[0][1:-1].detach().tolist()
    agg_gradient = []
    for begin_end in begin_end_list:
        agg_gradient.append(np.mean(gradient[begin_end[0]:begin_end[1]]))
    ranked = np.argsort(agg_gradient)
    largest_indices = ranked[::-1][:len(ranked)].tolist()
    return result_token_list, largest_indices

def process_data(input_data, key, output_file, tokenizer, model, embeddings, model_name):
    f = open(output_file, 'w')
    count = 0
    for index in tqdm(range(len(input_data[key]))):
        if len(input_data[key][index]['text'].split()) > 60:
            count += 1
            continue
        text = input_data[key][index]['text']
        label = input_data[key][index]['label']
        result_token_list, largest_indices = get_independent_token_and_rank(text, label, tokenizer, model, embeddings, model_name)
        if model_name == 'bert-base-uncased' or model_name == 'albert-base-v1' or model_name == 'xlnet-base-cased':
            result_text = " ".join(result_token_list)
        else:
            result_text = "".join(result_token_list)
        # import ipdb; ipdb.set_trace()
        item = {
            'text': result_text,
            'label': label,
            'rank': largest_indices
        }
        f.write(json.dumps(item) + '\n')
    print(count)
    f.close()

def main(args):
    dataset, num_labels, id_2_label = load_data(args.dataset, 'roberta')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    model.load_state_dict(torch.load(args.model_dir))
    # import ipdb; ipdb.set_trace()
    embeddings = model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long())
    model = model.cuda()
    
    # process_data(dataset, 'train', args.train_output, tokenizer, model, embeddings)
    process_data(dataset, 'validation', args.validation_output, tokenizer, model, embeddings, args.model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SST')
    parser.add_argument('--dataset', type=str, default='ag_news')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--model_dir', type=str, default='logdir/agnews/pytorch_model.bin')
    parser.add_argument('--train_output', type=str, default='data/ag_news/train_processed1.json')
    parser.add_argument('--validation_output', type=str, default='data/ag_news/test_processed2.json')
    args = parser.parse_args()

    main(args)
