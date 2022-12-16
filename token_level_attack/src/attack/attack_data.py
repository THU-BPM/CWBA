import copy
from curses.ascii import isalnum, isalpha, isdigit
from typing import OrderedDict
from src.utils import ids_to_word_begin_end_list

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
def get_proper_tokenization_bert(input_ids, label_ids, tokenizer,suffix_id_list):
    '''split a word to many parts,so the middle part can be changed'''
    '''reconstruct a entity token in three parts: prefix middle suffix,labels also need to be extended to match'''
    suffixs={tokenizer.convert_ids_to_tokens(id)[2:] for id in suffix_id_list}
    input_word = tokenizer.decode(input_ids)
    # if check_word(input_word) == False:
    #     return False, [], []
    # single ch,duplicate and the last changable
    if len(input_word)==1:
        id1=tokenizer.convert_tokens_to_ids(input_word[0])
        id2=tokenizer.convert_tokens_to_ids('##'+input_word[0])
        new_ids=[id1,id2]
        label_ids.append(-100)
        return new_ids,label_ids
    # two ch, the last changable
    elif len(input_word)==2:
        id1=tokenizer.convert_tokens_to_ids(input_word[0])
        id2=tokenizer.convert_tokens_to_ids('##'+input_word[1])
        # [UNK]
        if id2==100:
            id2=tokenizer.convert_tokens_to_ids('##'+input_word[0])
        new_ids=[id1,id2]
        while(len(label_ids)<2):
            label_ids.append(-100)
        return new_ids,label_ids
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
        # label id may be shorter,eg '2 / 6 / 2020'
        if len(new_ids)>=len(input_ids):
            label_ids = label_ids + [-100 for _ in range(len(new_ids) - len(input_ids))]
        else:
            label_ids=[label_ids[i] for i in range(len(new_ids))]
        return new_ids, label_ids
def get_proper_tokenization_roberta(input_ids, label_ids, tokenizer,suffix_id_list):
    '''split a word to many parts,so the middle part can be changed'''
    '''reconstruct a entity token in three parts: prefix middle suffix,labels also need to be extended to match'''
    UNK=3
    suffixs={tokenizer.convert_ids_to_tokens(id) for id in suffix_id_list}
    input_word = tokenizer.decode(input_ids)
    assert input_word[0]==' '
    input_word="Ġ"+input_word[1:]
    # single ch,duplicate and the last changable
    if (len(input_word)-1)==1:
        id1=tokenizer.convert_tokens_to_ids(input_word[:2])
        id2=tokenizer.convert_tokens_to_ids(input_word[1])
        new_ids=[id1,id2]
        label_ids.append(-100)
        return new_ids,label_ids
    # two chs, the last changable
    elif (len(input_word)-1)==2:
        id1=tokenizer.convert_tokens_to_ids(input_word[:2])
        id2=tokenizer.convert_tokens_to_ids(input_word[2])
        # [UNK]
        if id2==UNK:
            id2=tokenizer.convert_tokens_to_ids(input_word[1])
        new_ids=[id1,id2]
        while(len(label_ids)<2):
            label_ids.append(-100)
        return new_ids,label_ids
    else:
        search_length=len(input_word)//2
        prefix_length = -1
        suffix_length = -1
        ### prefix search
        for length in range(search_length, 0, -1):
            # eg:baltimore,ba can be encoded as a id
            if tokenizer.convert_tokens_to_ids(input_word[:length])!=UNK:
                prefix_length = length
                break
        
        if prefix_length == search_length and len(input_word) % 2 == 0:
            search_length -= 1
        ### suffix search
        for length in range(search_length, 0, -1):
            # print(input_word[len(input_word) - length -1:])
            if tokenizer.convert_tokens_to_ids(input_word[len(input_word) - length:]) != UNK:
                suffix_length = length
                break
        begin = [tokenizer.convert_tokens_to_ids(input_word[:prefix_length])]

        middle_str=input_word[prefix_length: len(input_word) - suffix_length]
        middle_strs=split_str(middle_str,suffixs)
        middle=[tokenizer.convert_tokens_to_ids(subtoken) for subtoken in middle_strs]

        end = [tokenizer.convert_tokens_to_ids(input_word[len(input_word) - suffix_length : ])]
        new_ids = begin + middle + end
        # label id may be shorter,eg '2 / 6 / 2020'
        if len(new_ids)>=len(input_ids):
            label_ids = label_ids + [-100 for _ in range(len(new_ids) - len(input_ids))]
        else:
            label_ids=[label_ids[i] for i in range(len(new_ids))]
        return new_ids, label_ids
def get_proper_tokenization_gpt(input_ids, label_ids, tokenizer,suffix_id_list):
    '''split a word to many parts,so the middle part can be changed'''
    '''reconstruct a entity token in three parts: prefix middle suffix,labels also need to be extended to match'''
    UNK=50256
    suffixs={tokenizer.convert_ids_to_tokens(id) for id in suffix_id_list}
    input_word = tokenizer.decode(input_ids)
    assert input_word[0]==' '
    input_word="Ġ"+input_word[1:]
    # single ch,duplicate and the last changable
    if (len(input_word)-1)==1:
        id1=tokenizer.convert_tokens_to_ids(input_word[:2])
        id2=tokenizer.convert_tokens_to_ids(input_word[1])
        new_ids=[id1,id2]
        label_ids.append(-100)
        return new_ids,label_ids
    # two chs, the last changable
    elif (len(input_word)-1)==2:
        id1=tokenizer.convert_tokens_to_ids(input_word[:2])
        id2=tokenizer.convert_tokens_to_ids(input_word[2])
        # [UNK]
        if id2==UNK:
            id2=tokenizer.convert_tokens_to_ids(input_word[1])
        new_ids=[id1,id2]
        while(len(label_ids)<2):
            label_ids.append(-100)
        return new_ids,label_ids
    else:
        search_length=len(input_word)//2
        prefix_length = -1
        suffix_length = -1
        ### prefix search
        for length in range(search_length, 0, -1):
            # eg:baltimore,ba can be encoded as a id
            if tokenizer.convert_tokens_to_ids(input_word[:length])!=UNK:
                prefix_length = length
                break
        
        if prefix_length == search_length and len(input_word) % 2 == 0:
            search_length -= 1
        ### suffix search
        for length in range(search_length, 0, -1):
            # print(input_word[len(input_word) - length -1:])
            if tokenizer.convert_tokens_to_ids(input_word[len(input_word) - length:]) != UNK:
                suffix_length = length
                break
        begin = [tokenizer.convert_tokens_to_ids(input_word[:prefix_length])]

        middle_str=input_word[prefix_length: len(input_word) - suffix_length]
        middle_strs=split_str(middle_str,suffixs)
        middle=[tokenizer.convert_tokens_to_ids(subtoken) for subtoken in middle_strs]

        end = [tokenizer.convert_tokens_to_ids(input_word[len(input_word) - suffix_length : ])]
        new_ids = begin + middle + end
        # label id may be shorter,eg '2 / 6 / 2020'
        if len(new_ids)>=len(input_ids):
            label_ids = label_ids + [-100 for _ in range(len(new_ids) - len(input_ids))]
        else:
            label_ids=[label_ids[i] for i in range(len(new_ids))]
        return new_ids, label_ids
def get_proper_tokenization_albert(input_ids, label_ids, tokenizer,suffix_id_list):
    '''split a word to many parts,so the middle part can be changed'''
    '''reconstruct a entity token in three parts: prefix middle suffix,labels also need to be extended to match'''
    UNK=1
    suffixs={tokenizer.convert_ids_to_tokens(id) for id in suffix_id_list}
    input_word = tokenizer.decode(input_ids)
    input_word="▁"+input_word
    # single ch,duplicate and the last changable
    if (len(input_word)-1)==1:
        id1=tokenizer.convert_tokens_to_ids(input_word[:2])
        if id1==UNK:
            id1=tokenizer.convert_tokens_to_ids("▁")
        id2=tokenizer.convert_tokens_to_ids(input_word[1])
        new_ids=[id1,id2]
        label_ids.append(-100)
        return new_ids,label_ids
    # two chs, the last changable
    elif (len(input_word)-1)==2:
        id1=tokenizer.convert_tokens_to_ids(input_word[:2])
        if id1==UNK:
            id1=tokenizer.convert_tokens_to_ids("▁")
        id2=tokenizer.convert_tokens_to_ids(input_word[2])
        # [UNK]
        if id2==UNK:
            id2=tokenizer.convert_tokens_to_ids(input_word[1])
        new_ids=[id1,id2]
        while(len(label_ids)<2):
            label_ids.append(-100)
        return new_ids,label_ids
    else:
        search_length=len(input_word)//2
        prefix_length = -1
        suffix_length = -1
        ### prefix search
        for length in range(search_length, 0, -1):
            # eg:baltimore,ba can be encoded as a id
            if tokenizer.convert_tokens_to_ids(input_word[:length])!=UNK:
                prefix_length = length
                break
        
        if prefix_length == search_length and len(input_word) % 2 == 0:
            search_length -= 1
        ### suffix search
        for length in range(search_length, 0, -1):
            # print(input_word[len(input_word) - length -1:])
            if tokenizer.convert_tokens_to_ids(input_word[len(input_word) - length:]) != UNK:
                suffix_length = length
                break
        begin = [tokenizer.convert_tokens_to_ids(input_word[:prefix_length])]

        middle_str=input_word[prefix_length: len(input_word) - suffix_length]
        middle_strs=split_str(middle_str,suffixs)
        middle=[tokenizer.convert_tokens_to_ids(subtoken) for subtoken in middle_strs]

        end = [tokenizer.convert_tokens_to_ids(input_word[len(input_word) - suffix_length : ])]
        new_ids = begin + middle + end
        # label id may be shorter,eg '2 / 6 / 2020'
        if len(new_ids)>=len(input_ids):
            label_ids = label_ids + [-100 for _ in range(len(new_ids) - len(input_ids))]
        else:
            label_ids=[label_ids[i] for i in range(len(new_ids))]
        return new_ids, label_ids
def get_proper_tokenization_xlnet(input_ids, label_ids, tokenizer,suffix_id_list):
    '''split a word to many parts,so the middle part can be changed'''
    '''reconstruct a entity token in three parts: prefix middle suffix,labels also need to be extended to match'''
    UNK=0
    suffixs={tokenizer.convert_ids_to_tokens(id) for id in suffix_id_list}
    input_word = tokenizer.decode(input_ids)
    input_word="▁"+input_word
    # single ch,duplicate and the last changable
    if (len(input_word)-1)==1:
        id1=tokenizer.convert_tokens_to_ids(input_word[:2])
        if id1==UNK:
            id1=tokenizer.convert_tokens_to_ids("▁")
        id2=tokenizer.convert_tokens_to_ids(input_word[1])
        new_ids=[id1,id2]
        while(len(label_ids)<2):
            label_ids.append(-100)
        return new_ids,label_ids
    # two chs, the last changable
    elif (len(input_word)-1)==2:
        id1=tokenizer.convert_tokens_to_ids(input_word[:2])
        if id1==UNK:
            id1=tokenizer.convert_tokens_to_ids("▁")
        id2=tokenizer.convert_tokens_to_ids(input_word[2])
        # [UNK]
        if id2==UNK:
            id2=tokenizer.convert_tokens_to_ids(input_word[1])
        new_ids=[id1,id2]
        while(len(label_ids)<2):
            label_ids.append(-100)
        return new_ids,label_ids
    else:
        search_length=len(input_word)//2
        prefix_length = -1
        suffix_length = -1
        ### prefix search
        for length in range(search_length, 0, -1):
            # eg:baltimore,ba can be encoded as a id
            if tokenizer.convert_tokens_to_ids(input_word[:length])!=UNK:
                prefix_length = length
                break
        
        if prefix_length == search_length and len(input_word) % 2 == 0:
            search_length -= 1
        ### suffix search
        for length in range(search_length, 0, -1):
            # print(input_word[len(input_word) - length -1:])
            if tokenizer.convert_tokens_to_ids(input_word[len(input_word) - length:]) != UNK:
                suffix_length = length
                break
        begin = [tokenizer.convert_tokens_to_ids(input_word[:prefix_length])]

        middle_str=input_word[prefix_length: len(input_word) - suffix_length]
        middle_strs=split_str(middle_str,suffixs)
        middle=[tokenizer.convert_tokens_to_ids(subtoken) for subtoken in middle_strs]

        end = [tokenizer.convert_tokens_to_ids(input_word[len(input_word) - suffix_length : ])]
        new_ids = begin + middle + end
        # label id may be shorter,eg '2 / 6 / 2020'
        if len(new_ids)>=len(input_ids):
            label_ids = label_ids + [-100 for _ in range(len(new_ids) - len(input_ids))]
        else:
            label_ids=[label_ids[i] for i in range(len(new_ids))]
        return new_ids, label_ids
def get_proper_tokenization(input_ids, label_ids, tokenizer,suffix_id_list):
    '''split a word to many parts,so the middle part can be changed'''
    '''reconstruct a entity token in three parts: prefix middle suffix,labels also need to be extended to match'''
    new_ids,new_label_ids=[],[]
    if tokenizer.name_or_path=='bert-base-uncased':
        new_ids,new_label_ids=get_proper_tokenization_bert(input_ids, label_ids, tokenizer,suffix_id_list)
    elif tokenizer.name_or_path=='gpt2-medium':
        new_ids,new_label_ids=get_proper_tokenization_gpt(input_ids, label_ids, tokenizer,suffix_id_list)
    elif tokenizer.name_or_path=='roberta-base':
        new_ids,new_label_ids=get_proper_tokenization_roberta(input_ids, label_ids, tokenizer,suffix_id_list)
    elif tokenizer.name_or_path=='albert-base-v1':
        new_ids,new_label_ids=get_proper_tokenization_albert(input_ids, label_ids, tokenizer,suffix_id_list)
    elif tokenizer.name_or_path=='xlnet-base-cased':
        new_ids,new_label_ids=get_proper_tokenization_xlnet(input_ids, label_ids, tokenizer,suffix_id_list)
    else:
        raise Exception("model name problem")
    return new_ids,new_label_ids

def get_attack_word_info_list(sorted_attack_word_ids, tokenizer, input_ids,offset=1):
    '''return dict list,dict contain word id and its begin_end in subword ids'''
    word_begin_end_list=ids_to_word_begin_end_list(tokenizer,input_ids)
    attack_word_info_list=[]
    for id in sorted_attack_word_ids:
        word_info=dict(word_index=id,begin_end=word_begin_end_list[id])
        attack_word_info_list.append(word_info)
    return attack_word_info_list
def get_attack_word(entity_info,tokens,total=1):
    def check_pass(id):
        token=tokens[id]
        for ch in token:
            if not isalnum(ch):
                return False
        return True
    res=[]
    attack_word_ids=entity_info['gradient_rank']
    for id in attack_word_ids:
        if check_pass(id):
            total-=1
            res.append(id)
            if total==0:
                break
    return res 

def entity_char_attack(example, label_to_id, tokenizer,suffix_id_list):
    if 'gpt' in tokenizer.name_or_path:
        offset=0
    else:
        offset=1
    label_list, input_ids, all_entity_info_list, all_tokens \
    = example['labels'], example['input_ids'],example['entity_info_list'],example['tokens']
    
    result_input_ids, result_label_list, gradient_mask_list, label_mask_list, \
    raw_sentences, token_type_ids, attention_mask, origin_labels ,entity_start_ends,ner_tags,new_all_entity_info_list = [], [], [], [], [], [], [], [],[],[],[]

    for idx in range(len(label_list)):

        for each_entity_attack_word_num in [1,2]:

            _input_ids, _label_list,tokens = input_ids[idx], label_list[idx],all_tokens[idx]
            entity_info_list=copy.deepcopy(all_entity_info_list[idx])
            attack_word_belongs_to_which_entity_map=dict()
            already_contain_words=[]
            # get attack word ids
            for i,entity_info in enumerate(entity_info_list):
                attack_word_ids=get_attack_word(entity_info,tokens,each_entity_attack_word_num)
                # all ids have been chosed,then allow attack same
                entity_info['attack_word_ids']=attack_word_ids
                for word_id in attack_word_ids:
                    if word_id not in attack_word_belongs_to_which_entity_map:
                        attack_word_belongs_to_which_entity_map[word_id]=[]
                    # map to the entity id in entity_info_list, starts from 0
                    attack_word_belongs_to_which_entity_map[word_id].append(i)
                already_contain_words.extend(attack_word_ids)
            # to make it substitute from left to right,may be have duplicate words,use set to remove
            sorted_attack_word_ids=sorted(list(set(already_contain_words)))

            # get attack word begin end
            attack_word_info_list= get_attack_word_info_list(sorted_attack_word_ids,tokenizer,_input_ids,offset)
            new_ids=[]
            new_labels=[]
            pre_bound=0
            # for each word that we want to change
            for word_info in attack_word_info_list:
                begin_end=word_info['begin_end']
                # extended word
                new_entity_ids, new_entity_label_ids = get_proper_tokenization(_input_ids[begin_end[0]: begin_end[1]], _label_list[begin_end[0]: begin_end[1]], tokenizer,suffix_id_list)
                # get segment from origin, origin[last entity end, current enttity begin)
                new_ids+=_input_ids[pre_bound:begin_end[0]]
                new_labels+=_label_list[pre_bound:begin_end[0]]

                new_begin=len(new_ids)

                # use new entity ids
                new_ids+=new_entity_ids
                new_labels+=new_entity_label_ids

                new_end=len(new_ids)

                pre_bound=begin_end[1]
                entity_ids=attack_word_belongs_to_which_entity_map[word_info['word_index']]
                for entity_id in entity_ids:
                    if 'attack_begin_end_list' not in entity_info_list[entity_id]:
                        entity_info_list[entity_id]['attack_begin_end_list']=[]
                    entity_info_list[entity_id]['attack_begin_end_list'].append((new_begin,new_end))
            # add origin's last segment
            new_ids+=_input_ids[pre_bound:]
            new_labels+=_label_list[pre_bound:]

            # add entity begin end info
            word_begin_end_list=ids_to_word_begin_end_list(tokenizer,new_ids)
            for entity_info in entity_info_list:
                entity_info['entity_begin_end']=word_begin_end_list[entity_info['index_in_words']]
            new_sequence_length=len(new_ids)
            # mask middle extended entity to be 0,and it will be searched
            gradient_mask=[1]*new_sequence_length
            for entity_info in entity_info_list:
                for begin,end in entity_info['attack_begin_end_list']:
                    entity_ids_length=end-begin
                    if entity_ids_length==2:
                        gradient_mask[begin+1]=0
                    else:
                        for i in range(begin+1,end-1):
                            gradient_mask[i]=0 
            # noise label only care changing the entity's begin subtoken to O_label,others are ignored, not in use
            noise_labels=[-100]*new_sequence_length
            # for begin,end in new_entity_begin_end_list:
            #     noise_labels[begin]=O_label
            # label_mask is entity mask, not in use
            new_entity_begin_end_list=[]
            label_mask=[1]*new_sequence_length      
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
            new_all_entity_info_list.append(entity_info_list)
    # entity_info_list add attack_begin_end_list and entity_begin_end
    return {'new_input_ids': result_input_ids, 'new_label_list': result_label_list,'entity_info_list':new_all_entity_info_list,
     'gradient_mask': gradient_mask_list, 'label_mask': label_mask_list,
     'raw_sentences': raw_sentences, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
    'origin_labels': origin_labels,'entity_start_ends':entity_start_ends,"ner_tags":ner_tags}


def context_change(example):
    pass