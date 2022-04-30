from PIL import Image
import os
import sys


def generate():
    for filename in os.listdir('train'):
        # don't edit already edited photos
        if len(filename.split("_")) > 2:
            pass
        img = Image.open('train/' + filename)
        hoz_flip = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        ver_flip = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        hoz_flip.save('train/' + filename.split('.')[0] + '_hoz_flip.jpg')
        ver_flip.save('train/' + filename.split('.')[0] + '_ver_flip.jpg')

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