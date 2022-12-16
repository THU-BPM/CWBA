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
from src.token_dataset import  tokenize_and_align_labels
from src.utils import  print_args, setup_seed
from src.models import get_model
import os
from datasets import load_dataset, load_metric

from datasets import load_dataset
def get_labels(labels_list):
    all_labels=set()
    for labels in labels_list:
        all_labels=all_labels | set(labels)
    all_labels=list(all_labels)
    all_labels.sort()
    return all_labels

def evaluate_f1(id_to_label,labels_list,predictions):
    true_predictions=[]
    true_labels_list=[]
    assert len(predictions)==len(labels_list)
    for prediction,labels in zip(predictions,labels_list):
        assert len(prediction)==len(labels)
        true_prediction=[]
        true_labels=[]
        for i,label in enumerate(labels):
            if label!=-100:
                true_labels.append(id_to_label[label])
                true_prediction.append(id_to_label[prediction[i]])
        true_predictions.append(true_prediction)
        true_labels_list.append(true_labels)

    metric = load_metric("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels_list,scheme='IOB2',mode='strict')
    return results

def main(args):
    setup_seed(777)
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # load dataset
    # datasets, label_to_id, id_to_label, num_labels = load_data(args.data_folder)
    data_files = {
        'adv': os.path.join('data', args.data_folder, f'adv_texts_{args.model}.json'),
    }
    eval_key='adv'
    datasets = load_dataset('json', data_files=data_files)
    if args.model=='bert-base-uncased':
        label_to_id_file='label_to_id.json'
    elif args.model=='roberta-base':
        label_to_id_file='label_to_id_roberta.json'
    elif args.model=='gpt2-medium':
        label_to_id_file='label_to_id_gpt.json'
    elif args.model=='albert-base-v1':
        label_to_id_file='label_to_id_albert.json'
    elif args.model=='xlnet-base-cased':
        label_to_id_file='label_to_id_xlnet.json'
    else:
        raise Exception('model name error')
    with open(os.path.join('data', args.data_folder, label_to_id_file),'r') as f:
        label_to_id = json.load(f)
    id_to_label = {id: label for label,id in label_to_id.items()}
    num_labels=len(label_to_id)
    # Load tokenizer, model
    tokenizer, model = get_model(args.model, num_labels, args.model_checkpoint_folder)
    
    ### get id input
    encoded_dataset = datasets.map(tokenize_and_align_labels, batched=True, fn_kwargs = {'tokenizer': tokenizer, 'label_to_id': label_to_id}, load_from_cache_file =False)
    total_example=len(encoded_dataset[eval_key]['tokens'])
    labels=encoded_dataset[eval_key]['labels']
    predictions=[]
    with torch.no_grad():
        model.eval()
        for i in range(0,total_example):
            # end=min(begin+batch_size,total_example)
            # all_entity=encoded_dataset[eval_key][i]
            input_ids=torch.tensor(encoded_dataset[eval_key]['input_ids'][i],device=device,dtype=torch.int64).unsqueeze(0)
            # token_type_ids=torch.tensor(encoded_dataset[eval_key]['token_type_ids'][begin:end],device=device,dtype=torch.int64)
            # attention_mask=torch.tensor(encoded_dataset[eval_key]['attention_mask'][begin:end],device=device,dtype=torch.int64)
            # predictions=model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
            prediction=model(input_ids=input_ids).logits
            prediction=torch.argmax(prediction[0],dim=1).tolist()
            predictions.append(prediction)
    output_path=os.path.join('data', args.data_folder, f'result_{args.model}.json')
    results=evaluate_f1(id_to_label,labels,predictions)
    output_results= {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
    print(results)
    with open(output_path,'w') as f:
        json.dump(output_results,f,indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")

    # save and load
    parser.add_argument("--output_name", default="", type=str,
                        help="specify name")
    parser.add_argument("--model_checkpoint_folder", default='logdir/atis-demo1/pytorch_model.bin', type=str,
                        help="folder for loading GPT2 model trained with BERT tokenizer")    
    parser.add_argument("--data_folder", type=str, default="atis_temp",
                        help="folder in which to store data")
    parser.add_argument("--model", default="bert-base-uncased", type=str,
                        help="type of model")
    parser.add_argument("--run_id", default=0, type=int,
                        help="run_id_for_save")
    args = parser.parse_args()
    print_args(args)
    start=time.time()
    main(args)
    print(f"Lasted time:{(time.time()-start)/60} min")