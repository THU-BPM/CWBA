from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
def convert_string_to_image(string="hello"):
    width, height = 150, 75
    img = Image.new('RGB', (150, 75), color=(255,255,255))
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype("font/ms.ttf", 40)
    text_width, text_height = d.textsize('hello', font=fnt)

    print(text_width, text_height)
    d.text(((width-text_width)/2, (height-text_height)/2), string, fill=(0, 0, 0), font=fnt)

    img.save('images/{}.png'.format(string))

if __name__ == '__main__':
    convert_string_to_image()

