import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def get_independent_tokens(ids):
    '''eg: 'a&b'->'a' '&' 'b' '''
    token_list = []
    current_token_list = []
    for id in ids:
        token = tokenizer.convert_ids_to_tokens(id)
        if len(current_token_list) == 0:
            current_token_list.append(id)
        elif token.startswith('##'):
            current_token_list.append(id)
        else:
            token_list.append(current_token_list)
            current_token_list = [id]
    token_list.append(current_token_list)
    result_token_list = []
    for item in token_list:
        result_token_list.append(tokenizer.decode(item))
    return result_token_list
        
def get_append_label(current_label):
    if current_label == 'O':
        return current_label
    else:
        return 'I-'+ current_label[2:]

def process_file(input_name, output_name):
    mis_match_number = 0
    with open(input_name, 'r') as f:
        result = []
        data = f.readlines()
        for line in data:
            tokens, labels = [],[]
            item = json.loads(line)
            for index in range(len(item['tokens'])):
                token_ids = tokenizer.encode(item['tokens'][index], add_special_tokens=False)
                result_token_list = get_independent_tokens(token_ids)
                result_label_list = [item['ner_tags'][index]]
                append_label =get_append_label(item['ner_tags'][index])
                for _ in range(len(result_token_list) - 1):
                    result_label_list.append(append_label)
                tokens.extend(result_token_list)
                labels.extend(result_label_list)
            if len(tokens) != len(item['tokens']):
                # print(tokens)
                # print(item['tokens'])
                mis_match_number += 1
                # import ipdb; ipdb.set_trace()
            result.append({
                'tokens': tokens,
                'ner_tags': labels,
                'sentence': ' '.join(tokens)
            })
    print(input_name, mis_match_number, len(data))
    with open(output_name, 'w') as f:
        for item in result:
            f.write(json.dumps(item) + '\n')

process_file('data/atis/train_temp.json', 'data/atis/train_temp_processed.json')
process_file('data/atis/dev_temp.json', 'data/atis/dev_temp_processed.json')
process_file('data/atis/test_temp.json', 'data/atis/test_temp_processed.json')