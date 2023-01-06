import torch
from transformers import AutoTokenizer
import json
from transformers import AutoConfig, AutoModelForTokenClassification
from datasets import load_dataset
import argparse
import os


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

def get_label_info(data_folder):
    data_files = {
        'train': os.path.join('data', data_folder, 'train_temp.json'),
        'validation': os.path.join('data', data_folder, 'dev_temp.json')
    }
    extension = 'json'
    datasets = load_dataset(extension, data_files=data_files)
    all_labels = []
    all_labels.extend(datasets["train"]['ner_tags'])
    all_labels.extend(datasets["validation"]['ner_tags'])
    label_list = get_label_list(all_labels)
    id_to_label = {i: l for i, l in enumerate(label_list)}
    return label_list, id_to_label

def decode(ids, tokenizer):
    str_list = []
    for id in ids:
        str_list.append(tokenizer.decode(id))
    return str_list

def model_inference(model, tokenizer, text, id_to_label):
    text_input = tokenizer(text, max_length=256, truncation=True)
    pred = model(input_ids=torch.LongTensor(text_input['input_ids']).unsqueeze(0).cuda()).logits.data.cpu()
    pred_label = torch.argmax(pred[0], dim=1).tolist()
    str_list = decode(text_input['input_ids'], tokenizer)
    pred_label_str = [id_to_label[label] for label in pred_label]
    str_list_ann = [str_list[i] + '(' + pred_label_str[i] + ')' for i in range(len(pred_label_str))]
    return pred_label_str, str_list_ann
    
if __name__ == '__main__':
    '''
    python view_data.py --data_file adv_samples/attack1 --output_file sample/atis-run1/prefix1.json
    '''
    parser = argparse.ArgumentParser(description="White-box attack.")
    parser.add_argument("--data_file", default="adv_samples/attack1", type=str,
                        help="folder for find origin file")
    parser.add_argument("--output_file", default="sample/atis-run1/prefix1.json", type=str,
                        help="folder for output file")
    parser.add_argument("--pretrain_model", default="bert-base-uncased", type=str,
                        help="folder for pretrain model")
    parser.add_argument("--model_load_dir", default='logdir/atis-demo1/pytorch_model.bin', type=str,
                        help="model load dir")
    parser.add_argument("--data_folder", type=str, default="atis",
                        help="folder in which to store data")
    parser.add_argument("--tau", default=0.5, type=float,
                        help="tau for gumbel softmax")
    parser.add_argument("--run_id", default=0, type=int,
                        help="run_id_for_save")

    args = parser.parse_args()
    data = torch.load(args.data_file)
    label_list, id_to_label = get_label_info(args.data_folder)
    num_labels = len(label_list)
    config = AutoConfig.from_pretrained(
        args.pretrain_model,
        num_labels=num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrain_model,
        add_prefix_space=True
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.pretrain_model,
        config=config,
    ).cuda()

    model.load_state_dict(torch.load(args.model_load_dir))
     ## TODO: change get label method
    
    json_data = []
    for index in range(len(data['clean_texts'])):
        clean_text = data['clean_texts'][index]
        adv_text = data['adv_texts'][index]
        clean_pred_label_str, clean_str_list_ann = model_inference(model, tokenizer, clean_text, id_to_label)
        adv_pred_label_str, adv_str_list_ann = model_inference(model, tokenizer, adv_text, id_to_label)
        ## TODO: temporarily use try except
        try:
            adv_str_list_ann_contrastive = [adv_str_list_ann[i] + ' (' + clean_pred_label_str[i] + ')' for i in range(len(adv_str_list_ann))]
            json_data.append({
                'clean_text': clean_text,
                'adv_text': adv_text,
                # 'clean_pred_label': clean_pred_label_str,
                # 'adv_pred_label': adv_pred_label_str,
                # 'clean_str_list_ann':  clean_str_list_ann,
                # 'adv_str_list_ann': adv_str_list_ann,
                'adv_str_list_ann_contrastive': adv_str_list_ann_contrastive
            })
        except:
            pass
    output_file = "{}_{}_{}_{}".format(args.output_file, args.tau, args.run_id, '.json')
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=4)
