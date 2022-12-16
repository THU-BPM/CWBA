import copy
from selectors import EpollSelector
from typing import OrderedDict
import random

NOT_ATTACK_SENTENCES=[]
def get_entity_begin_label(label_to_id):
    return {id for label,id in label_to_id.items() if label.startswith('B-')}

def get_entity_begin_end(label_list, label_to_id):
    begin_end_list = []
    begin_label=get_entity_begin_label(label_to_id)
    for begin,label in enumerate(label_list):
        if label in begin_label:
            end=begin+1
            while(end<len(label_list) and label_list[end]==-100): end+=1
            begin_end_list.append((begin,end))
    return begin_end_list

def entity_around_insert(example, label_to_id, left_number, right_number):
    O_label = label_to_id['O']
    label_list, input_ids = example['labels'], example['input_ids']
    result_input_ids, result_label_list, gradient_mask_list, label_mask_list, raw_sentences, token_type_ids, attention_mask, origin_labels = [], [], [], [], [], [], [], []
    
    for idx in range(len(label_list)):
        _input_ids, _label_list = input_ids[idx], label_list[idx]
        begin_end_list = get_entity_begin_end(_label_list[1:-1], O_label)
        for begin_end in begin_end_list:
            origin_label = _label_list[:begin_end[0] + 1] + [O_label]*left_number + _label_list[begin_end[0] + 1: begin_end[1]+1] + [O_label]*right_number + _label_list[begin_end[1]+1: ]
            label_mask = [1 for _ in range(len(origin_label))]
            for index in range(begin_end[0] + 1, begin_end[1]+1):
                _label_list[index] = O_label
                label_mask[index + left_number] = 0
            new_ids = _input_ids[:begin_end[0] + 1] + [0]*left_number + _input_ids[begin_end[0] + 1: begin_end[1]+1] + [0]*right_number + _input_ids[begin_end[1]+1: ]
            new_labels = _label_list[:begin_end[0] + 1] + [O_label]*left_number + _label_list[begin_end[0] + 1: begin_end[1]+1] + [O_label]*right_number + _label_list[begin_end[1]+1: ]
            gradient_mask = [1 if id != 0 else 0 for id in new_ids]
            result_input_ids.append(new_ids)
            result_label_list.append(new_labels)
            gradient_mask_list.append(gradient_mask)
            label_mask_list.append(label_mask)
            token_type_ids.append(example['token_type_ids'][idx] + [0]*(left_number + right_number))
            raw_sentences.append(example['sentence'][idx])
            attention_mask.append(example['attention_mask'][idx] + [1]*(left_number + right_number))
            origin_labels.append(origin_label)

    return {'new_input_ids': result_input_ids, 'new_label_list': result_label_list, 'gradient_mask': gradient_mask_list, 'label_mask': label_mask_list, 'raw_sentences': raw_sentences, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'origin_labels': origin_labels}


def entity_around_change(example, label_to_id, left_number, right_number):
    O_label = label_to_id['O']
    label_list, input_ids = example['labels'], example['input_ids']
    result_input_ids, result_label_list, gradient_mask_list, label_mask_list, raw_sentences, token_type_ids, attention_mask, origin_labels = [], [], [], [], [], [], [], []
    for idx in range(len(label_list)):
        _input_ids, _label_list = input_ids[idx], label_list[idx]
        begin_end_list = get_entity_begin_end(_label_list[1:-1], O_label)
        for begin_end in begin_end_list:
            origin_label = _label_list[:begin_end[0] + 1] + _label_list[begin_end[0] + 1: begin_end[1]+1] + _label_list[begin_end[1]+1: ]
            label_mask = [1 for _ in range(len(origin_label))]
            for index in range(begin_end[0] + 1, begin_end[1]+1):
                _label_list[index] = O_label
                label_mask[index] = 0
            new_left_number = min(begin_end[0] + 1, left_number)
            new_right_number = min(len(_input_ids) - begin_end[1] -  2, right_number)
            new_ids = _input_ids[:begin_end[0] + 1 - new_left_number] + [0]*new_left_number + _input_ids[begin_end[0] + 1: begin_end[1]+1] + [0]*new_right_number + _input_ids[begin_end[1]+1 + new_right_number: ]
            new_labels = _label_list[:begin_end[0] + 1] + _label_list[begin_end[0] + 1: begin_end[1]+1] + _label_list[begin_end[1]+1: ]
            gradient_mask = [1 if id != 0 else 0 for id in new_ids]
            result_input_ids.append(_input_ids)
            result_label_list.append(new_labels)
            gradient_mask_list.append(gradient_mask)
            label_mask_list.append(label_mask)
            token_type_ids.append(example['token_type_ids'][idx])
            raw_sentences.append(example['sentence'][idx])
            attention_mask.append(example['attention_mask'][idx])
            origin_labels.append(origin_label)
    return {'new_input_ids': result_input_ids, 'new_label_list': result_label_list, 'gradient_mask': gradient_mask_list, 'label_mask': label_mask_list, 'raw_sentences': raw_sentences, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'origin_labels': origin_labels}

def split_str(string,suffixs):
    '''split string to subtoken that is in suffixs'''
    middle=[]
    _string=string
    while(string):
        idx=len(string)
        while(idx>0 and string[:idx] not in suffixs):
            idx-=1
        # alpha must in suffixs,idx=0 mean the first is not a alpha, just skip it,
        if idx==0:
            string=string[1:]
            continue
        middle.append(string[:idx])
        string=string[idx:]
    return middle

def check_word(string):
    # TODO 
    if len(string)<6 or not str.isalpha(string):
        return False
def get_proper_tokenization_bert(input_word, tokenizer,suffix_id_list):
    '''reconstruct a entity token in three parts: prefix middle suffix,labels also need to be extended to match'''
    suffixs={tokenizer.convert_ids_to_tokens(id)[2:] for id in suffix_id_list}

    # single ch,duplicate and the last changable
    if len(input_word)==1:
        id1=tokenizer.convert_tokens_to_ids(input_word[0])
        id2=tokenizer.convert_tokens_to_ids('##'+input_word[0])
        new_ids=[id1,id2]
        return new_ids
    # two ch, the last changable
    elif len(input_word)==2:
        id1=tokenizer.convert_tokens_to_ids(input_word[0])
        id2=tokenizer.convert_tokens_to_ids('##'+input_word[1])
        # [UNK]
        if id2==100:
            id2=tokenizer.convert_tokens_to_ids('##'+input_word[0])
        new_ids=[id1,id2]
        return new_ids
    else:
        search_length=len(input_word)//2
        prefix_length = -1
        suffix_length = -1
        ### prefix search
        for length in range(search_length, 0, -1):
            # eg:baltimore,ba can be encoded as a id
            if len(tokenizer.encode(input_word[:length], add_special_tokens=False)) == 1:
                prefix_length = length
                break
        
        if prefix_length == search_length and len(input_word) % 2 == 0:
            search_length -= 1
    
        ### suffix search
        for length in range(search_length, 0, -1):
            # print(input_word[len(input_word) - length -1:])
            if tokenizer.convert_tokens_to_ids('##' + input_word[len(input_word) - length:]) != 100:
                suffix_length = length
                break
        begin = tokenizer.encode(input_word[:prefix_length], add_special_tokens=False)

        middle_str=input_word[prefix_length: len(input_word) - suffix_length]
        middle_strs=split_str(middle_str,suffixs)
        middle=[tokenizer.convert_tokens_to_ids("##"+subtoken) for subtoken in middle_strs]

        end = [tokenizer.convert_tokens_to_ids( '##' +  input_word[len(input_word) - suffix_length: ])]
        new_ids = begin + middle + end
        return new_ids

def get_proper_tokenization_roberta(input_word, tokenizer,suffix_id_list, text, model):
    '''reconstruct a entity token in three parts: prefix middle suffix,labels also need to be extended to match'''
    # must begin with space
    if input_word[0] != " " or len(input_word) <= 1:
        import ipdb; ipdb.set_trace()
    assert input_word[0] == " " and len(input_word) > 1
    unk = 3 if model == 'roberta-base' else 50256
    suffixs={tokenizer.convert_ids_to_tokens(id) for id in suffix_id_list}
    
    # single ch,duplicate and the last changable
    if len(input_word)==2:
        id1=tokenizer.convert_tokens_to_ids("Ġ" + input_word[1])
        id2=tokenizer.convert_tokens_to_ids(input_word[1:])
        new_ids=[id1,id2]
        return new_ids
    # two ch, the last changable
    elif len(input_word)==3:
        
        id1=tokenizer.convert_tokens_to_ids("Ġ" + input_word[1])
        id2=tokenizer.convert_tokens_to_ids(input_word[2:])
        if id2==3 or id1 ==3:
            import ipdb; ipdb.set_trace()
        assert id2 !=3 and id1 !=3
        # [UNK]
        # if id2==100:
        #     id2=tokenizer.convert_tokens_to_ids('##'+input_word[0])
        new_ids=[id1,id2]
        return new_ids
    else:
        search_length=(len(input_word)-1)//2
        prefix_length = -1
        suffix_length = -1
        ### prefix search
        for length in range(search_length, 0, -1):
            # eg:baltimore,ba can be encoded as a id
            if len(tokenizer.encode(input_word[:length + 1], add_special_tokens=False)) == 1:
                prefix_length = length
                break
       
        if prefix_length == search_length and (len(input_word)-1) % 2 == 0:
            search_length -= 1
        
        prefix_length += 1
    
        ### suffix search
        for length in range(search_length, 0, -1):
            # print(input_word[len(input_word) - length -1:])
            if tokenizer.convert_tokens_to_ids(input_word[len(input_word) - length:]) != unk:
                suffix_length = length
                break
        
        begin = tokenizer.encode(input_word[:prefix_length], add_special_tokens=False)

        middle_str=input_word[prefix_length: len(input_word) - suffix_length]
        
        middle_strs=split_str(middle_str,suffixs)
        middle=[tokenizer.convert_tokens_to_ids(subtoken) for subtoken in middle_strs]
        end = [tokenizer.convert_tokens_to_ids(input_word[len(input_word) - suffix_length: ])]
        new_ids = begin + middle + end
        # import ipdb; ipdb.set_trace()
        return new_ids


def get_proper_tokenization_albert(input_word, tokenizer,suffix_id_list, model):
    '''reconstruct a entity token in three parts: prefix middle suffix,labels also need to be extended to match'''
    # must begin with space
    # import ipdb; ipdb.set_trace()
    # assert input_word[0] == "▁" and len(input_word) > 0
    # input_word = "Ġ" + input_word[1:]
    unk = 0 if model == 'xlnet-base-cased' else 1
    suffixs={tokenizer.convert_ids_to_tokens(id) for id in suffix_id_list}
    
    # single ch,duplicate and the last changable
    if len(input_word)==1:
        id1=tokenizer.convert_tokens_to_ids("▁" + input_word[0])
        id2=tokenizer.convert_tokens_to_ids(input_word[0:])
        new_ids=[id1, id2]
        return new_ids
    # two ch, the last changable
    elif len(input_word)==2:
        
        id1=tokenizer.convert_tokens_to_ids("▁" + input_word[0])
        id2=tokenizer.convert_tokens_to_ids(input_word[1:])
        # if id2==1 or id1 ==1:
        #     import ipdb; ipdb.set_trace()
        # assert id2 !=1 and id1 !=1
        # [UNK]
        # if id2==100:
        #     id2=tokenizer.convert_tokens_to_ids('##'+input_word[0])
        new_ids=[id1,id2]
        return new_ids
    else:
        search_length=(len(input_word))//2
        prefix_length = -1
        suffix_length = -1
        ### prefix search
        for length in range(search_length, 0, -1):
            # eg:baltimore,ba can be encoded as a id
            if len(tokenizer.encode(input_word[:length], add_special_tokens=False)) == 1:
                prefix_length = length
                break
       
        if prefix_length == search_length and (len(input_word)) % 2 == 0:
            search_length -= 1

        ### suffix search
        for length in range(search_length, 0, -1):
            if tokenizer.convert_tokens_to_ids(input_word[len(input_word) - length:]) != unk:
                suffix_length = length
                break
        # if input_word.startswith("air"):
        #     import ipdb; ipdb.set_trace()
        begin = tokenizer.encode(input_word[:prefix_length], add_special_tokens=False)

        middle_str=input_word[prefix_length: len(input_word) - suffix_length]
        
        middle_strs=split_str(middle_str,suffixs)
        middle=[tokenizer.convert_tokens_to_ids(subtoken) for subtoken in middle_strs]
        end = [tokenizer.convert_tokens_to_ids(input_word[len(input_word) - suffix_length: ])]
        new_ids = begin + middle + end
        return new_ids


def entity_char_attack(example, label_to_id, tokenizer,suffix_id_list):
    O_label = label_to_id['O']
    label_list, input_ids = example['labels'], example['input_ids']
    result_input_ids, result_label_list, gradient_mask_list, label_mask_list, raw_sentences, token_type_ids, attention_mask, origin_labels ,entity_start_ends,ner_tags = [], [], [], [], [], [], [], [],[],[]
    for idx in range(len(label_list)):
        _input_ids, _label_list = input_ids[idx], label_list[idx]
        # a entity_word's begin and end idx.(a entity_word is a word in an entity, an entity
        # may consist of many word)
        # if idx!=253:continue
        begin_end_list = get_entity_begin_end(_label_list[1:-1], label_to_id)
        new_ids=[]
        new_labels=[]
        # [new_begin,new_end) is new entity position in extended ids
        new_entity_begin_end_list=[]
        pre_bound=0
        # for each word
        for begin_end in begin_end_list:
            # extended word
            new_entity_ids, new_entity_label_ids = get_proper_tokenization(_input_ids[begin_end[0] + 1: begin_end[1]+1], _label_list[begin_end[0] + 1: begin_end[1]+1], tokenizer,suffix_id_list)
            # get segment from origin, origin[last entity end, current enttity begin)
            new_ids+=_input_ids[pre_bound:begin_end[0]+1]
            new_labels+=_label_list[pre_bound:begin_end[0]+1]

            new_begin=len(new_ids)

            # use new entity ids
            new_ids+=new_entity_ids
            new_labels+=new_entity_label_ids

            new_end=len(new_ids)

            pre_bound=begin_end[1]+1
            new_entity_begin_end_list.append((new_begin,new_end))
        # add origin's last segment
        new_ids+=_input_ids[pre_bound:]
        new_labels+=_label_list[pre_bound:]

        # # no changable entity , skip the sentence 
        # if not has_valid_entity:
        #     global NOT_ATTACK_SENTENCES

        #     NOT_ATTACK_SENTENCES.append(OrderedDict(tokens=example['tokens'][idx],
        #                                             ner_tags=example['ner_tags'][idx],
        #     ))
        #     continue

        new_sequence_length=len(new_ids)
        # mask middle extended entity to be 0,and it will be searched
        gradient_mask=[1]*new_sequence_length
        for begin,end in new_entity_begin_end_list:
            entity_ids_length=end-begin
            if entity_ids_length==2:
                gradient_mask[begin+1]=0
            else:
                for i in range(begin+1,end-1):
                    gradient_mask[i]=0
        # noise label only care changing the entity's begin subtoken to O_label,others are ignored
        noise_labels=[-100]*new_sequence_length
        for begin,end in new_entity_begin_end_list:
            noise_labels[begin]=O_label
        # label_mask is entity mask
        label_mask=[1]*new_sequence_length
        for begin,end in new_entity_begin_end_list:
            for i in range(begin,end):
                label_mask[i]=0        
        result_input_ids.append(new_ids)
        result_label_list.append(noise_labels)
        gradient_mask_list.append(gradient_mask)
        label_mask_list.append(label_mask)
        token_type_ids.append([0]*new_sequence_length) 
        raw_sentences.append(example['sentence'][idx])
        attention_mask.append([1]*new_sequence_length)
        origin_labels.append(new_labels)
        entity_start_ends.append(new_entity_begin_end_list)
        ner_tags.append(example['ner_tags'][idx])
    return {'new_input_ids': result_input_ids, 'new_label_list': result_label_list,
     'gradient_mask': gradient_mask_list, 'label_mask': label_mask_list,
     'raw_sentences': raw_sentences, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
    'origin_labels': origin_labels,'entity_start_ends':entity_start_ends,"ner_tags":ner_tags}


def context_change(example):

    pass


def random_get_attack_token_index(independent_tokens, max_number):
    result_index = []
    while len(result_index) <= max_number:
        index = random.randint(0,len(independent_tokens) - 1)
        if index not in result_index and len(independent_tokens[index]) >= 3:
            result_index.append(index)
    return result_index

def grad_get_attack_token_index(independent_tokens, rank, max_number):
    result_index = []
    for rank_idx in rank:
        if len(independent_tokens[rank_idx].replace(' ', '')) >= 2:
            result_index.append(rank_idx)
        if len(result_index) >= max_number:
            break
    return result_index

def get_independent_tokens_bert(ids, tokenizer):
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


def get_independent_tokens_roberta(ids, tokenizer):
    '''eg: 'a&b'->'a' '&' 'b' '''
    token_list = []
    current_token_list = []
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
            current_token_list = [id]
    token_list.append(current_token_list)
    result_token_list = []
    for item in token_list:
        current_string = ""
        for id in item:
            current_string += tokenizer.convert_ids_to_tokens(id)
        current_string = current_string.replace('Ġ', ' ')
        result_token_list.append(current_string)
    return result_token_list


def get_independent_tokens_albert(ids, tokenizer):
    '''eg: 'a&b'->'a' '&' 'b' '''
    token_list = []
    current_token_list = []
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
            current_token_list = [id]
    token_list.append(current_token_list)
    result_token_list = []
    for item in token_list:
        result_token_list.append(tokenizer.decode(item))
    return result_token_list


def token_select_attack(example, tokenizer, suffix_id_list, mode, args):
    label_list, texts = example['label'], example['text']
    if mode == 'grad':
        rank = example['rank']
        
    result_input_ids, result_label_list, gradient_mask_list, attack_indexes, all_tokens = [], [], [], [], []
    for idx in range(len(label_list)):
        text, _label_list = texts[idx], label_list[idx]
        tokens = text.split()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if args.model == 'bert-base-uncased':
            independent_tokens = get_independent_tokens_bert(token_ids, tokenizer)
        elif args.model == 'roberta-base' or args.model == 'gpt2-medium':
            independent_tokens = get_independent_tokens_roberta(token_ids, tokenizer)
        else:
            independent_tokens = get_independent_tokens_albert(token_ids, tokenizer)

        for g_number in range(args.max_add):
            select_number = args.initial_select_number + g_number * args.per_add_number
            if mode != 'grad':
                current_rank = list(range(len(independent_tokens)))
                random.shuffle(current_rank)
            else:
                current_rank = rank[idx]
            
            attack_index = grad_get_attack_token_index(independent_tokens, current_rank, select_number)
            
            input_ids = []
            _gradient_mask_list = []
            # import ipdb; ipdb.set_trace()
            for index in range(len(independent_tokens)):
                if index in attack_index:
                    if args.model == 'bert-base-uncased':
                        ids = get_proper_tokenization_bert(independent_tokens[index],  tokenizer, suffix_id_list)
                    elif args.model == 'roberta-base' or args.model == 'gpt2-medium':
                        ids = get_proper_tokenization_roberta(independent_tokens[index],  tokenizer, suffix_id_list, text, args.model)
                    elif args.model == 'albert-base-v1' or args.model == 'xlnet-base-cased':
                        ids = get_proper_tokenization_albert(independent_tokens[index],  tokenizer, suffix_id_list, args.model)
                    # if independent_tokens[index].startswith("air"):
                    #     import ipdb; ipdb.set_trace()
                    # import ipdb; ipdb.set_trace()
                    # for id in ids:
                    #     print(tokenizer.convert_ids_to_tokens(id), id)
                    # print()
                    gradient_mask = [1] * len(ids)
                    _begin = 1
                    if len(ids) > 3:
                        _begin = 2
                    for pos in range(_begin, len(ids) - 1):
                        gradient_mask[pos] = 0
                else:
                    ids = tokenizer.encode(independent_tokens[index], add_special_tokens=False)
                    gradient_mask = [1] * len(ids)
                input_ids.extend(ids)
                _gradient_mask_list.extend(gradient_mask)
            # import ipdb; ipdb.set_trace()
            attack_indexes.append(attack_index)
            result_input_ids.append(input_ids)
            gradient_mask_list.append(_gradient_mask_list)

            result_label_list.append(_label_list)
            all_tokens.append(tokens)
    return {'new_input_ids': result_input_ids, 'new_label_list': result_label_list, 'gradient_mask': gradient_mask_list, 'attack_indexes': attack_indexes, 'all_tokens': all_tokens}
