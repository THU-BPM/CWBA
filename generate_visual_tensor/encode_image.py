from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def convert_image(input_string):
    width, height = 150, 75
    img = Image.new('RGB', (150, 75), color=(255,255,255))
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype("font/ms.ttf", 40)
    text_width, text_height = d.textsize(input_string, font=fnt)
    # print(text_width, text_height)
    d.text(((width-text_width)/2, (height-text_height)/2), input_string, fill=(0, 0, 0), font=fnt)
    img.save('images/{}.png'.format(input_string))

img2vec = Img2Vec(cuda=True, model='resnet50')

def get_vec_of_image(image_path):
    img = Image.open(image_path)
    vec = img2vec.get_vec(img, tensor=True)
    return vec


def calculate_sim(input_string1, input_string2):
    convert_image(input_string1)
    convert_image(input_string2)
    vec1 = get_vec_of_image('images/{}.png'.format(input_string1)).view(1, -1)
    vec2 = get_vec_of_image('images/{}.png'.format(input_string2)).view(1, -1)
    s = cosine_similarity(vec1, vec2)
    print('{}-{}-{}'.format(input_string1, input_string2, s))

calculate_sim('HH', 'h')
calculate_sim('afefeb', 'a9800b')
calculate_sim('Hh', 'h')
calculate_sim('hh', 'h')
calculate_sim('ll', '11')
calculate_sim('ah', 'ha')
calculate_sim('hello', 'he110')
calculate_sim('qwe', 'ewq')
calculate_sim('qwe', 'eqw')
calculate_sim('q', 'g')
calculate_sim('q', 'm')
calculate_sim('q', 'qq')
import ipdb; ipdb.set_trace()