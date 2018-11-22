from PIL import Image
import os, sys

path = "./small/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            width, height = im.size
            f, e = os.path.splitext(path+item)
            imResize = im.resize((int(width/2),int(height/2)), Image.ANTIALIAS)
            print(f)
            print(e)
            imResize.save(f + '.png', 'PNG')

resize()