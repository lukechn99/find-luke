from PIL import Image
import os
import sys
from filetype import is_image

def generate():
    for filename in os.listdir('train'):
        # don't edit already edited photos
        if len(filename.split("_")) > 2 or not is_image(filename):
            pass
        img = Image.open('train/' + filename)
        hoz_flip = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        ver_flip = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        rotate_30 = img.rotate(30)
        rotate_60 = img.rotate(60)
        rotate_120 = img.rotate(120)
        rotate_150 = img.rotate(150)
        rotate_210 = img.rotate(210)
        rotate_240 = img.rotate(240)
        rotate_300 = img.rotate(300)
        rotate_330 = img.rotate(330)
        hoz_flip.save('train/' + filename.split('.')[0] + '_hoz_flip.jpg')
        ver_flip.save('train/' + filename.split('.')[0] + '_ver_flip.jpg')
        rotate_30.save('train/' + filename.split('.')[0] + '_rotate_30.jpg')
        rotate_60.save('train/' + filename.split('.')[0] + '_rotate_60.jpg')
        rotate_120.save('train/' + filename.split('.')[0] + '_rotate_120.jpg')
        rotate_150.save('train/' + filename.split('.')[0] + '_rotate_150.jpg')
        rotate_210.save('train/' + filename.split('.')[0] + '_rotate_210.jpg')
        rotate_240.save('train/' + filename.split('.')[0] + '_rotate_240.jpg')
        rotate_300.save('train/' + filename.split('.')[0] + '_rotate_300.jpg')
        rotate_330.save('train/' + filename.split('.')[0] + '_rotate_330.jpg')


def clean():
    for filename in os.listdir('train'):
        if len(filename.split("_")) > 2:
            try:
                os.remove('train/' + filename)
            except OSError as e:
                print(e)

commands = {
    "clean": clean(),
    "generate": generate()
}

for args in sys.argv[1:]:
    commands[args]