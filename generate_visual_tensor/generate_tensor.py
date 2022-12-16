import torch
import transformers
from transformers import AutoTokenizer
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import json

def convert_image(input_string,filename='tmp.png'):
    width, height = 200, 75
    img = Image.new('RGB', (width, height), color=(255,255,255))
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype("font/helvetica.ttf", 40)
    text_width, text_height = d.textsize(input_string, font=fnt)
    d.text(((width-text_width)/2, (height-text_height)/2), input_string, fill=(0, 0, 0), font=fnt)
    img.save(filename)


def get_vec_of_image(image_path,img2vec):
    img = Image.open(image_path)
    vec = img2vec.get_vec(img, tensor=True)
    return vec
def get_suffix_vocab_list(tokenizer, adding_pad=False,model_name='bert-base-uncased'):
    words_info = [(tok, ids) for tok, ids in tokenizer.vocab.items()]
    words_info=sorted(words_info)
    suffix_id_list = []
    if adding_pad:
        suffix_id_list.append(tokenizer.encode('[PAD]')[1])
    if model_name=='bert-base-uncased':
        for word, id in words_info:
            if word.startswith('##'):
                suffix_id_list.append(id)
    elif model_name=='roberta-base' or model_name=='gpt2-medium':
        for word, id in words_info:
            if not word.startswith('Ġ'):
                suffix_id_list.append(id)
    elif model_name=='xlnet-base-cased' or model_name=='albert-base-v1':
        for word, id in words_info:
            if not word.startswith('▁'):
                suffix_id_list.append(id)
    else:
        raise Exception('model name error')
    return suffix_id_list
def main():
    img2vec = Img2Vec(cuda=True, model='resnet50')
    tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
    suffix_ids=get_suffix_vocab_list(tokenizer,adding_pad=False)
    suffixs=[tokenizer.convert_ids_to_tokens(id) for id in suffix_ids]
    import ipdb; ipdb.set_trace()
    result=dict()
    for id,suffix in zip(suffix_ids,suffixs):
        convert_image(suffix,'tmp.png')
        vec=get_vec_of_image('tmp.png',img2vec)
        result[id]=vec.squeeze()
    print(len(result))
    torch.save(result,"suffix_visual_tensor_bert-base-uncased.pth")
    
if __name__ == '__main__':
    s=time.time()
    main()
    print(f"Total time {(time.time()-s)/60}")