# Простая программа для обрезания изображений
from PIL import Image
import json


def main_resize(user_id):
    with open('data_file.json', 'r') as j:
        json_data = json.load(j)

    imsize = int(json_data[user_id]["im_size"])

    content_img = Image.open(user_id + "content.jpg")
    style_img = Image.open(user_id + "style.jpg")

    content_img = content_img.resize((imsize, imsize))
    style_img = style_img.resize((imsize, imsize))

    content_img.save(user_id + 'content_sqr.jpg')
    style_img.save(user_id + 'style_sqr.jpg')
