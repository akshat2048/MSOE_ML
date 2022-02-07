from ctypes import resize
from PIL import Image
import os

def resize_images(DIRECTORY):
    list_of_images = os.listdir(DIRECTORY)
    for image in list_of_images:
        if "._" in image:
            list_of_images.remove(image)
        if "png" not in image:
            list_of_images.remove(image)
    
    for file in list_of_images:
        img = Image.open(os.path.join(DIRECTORY, file))
        img.thumbnail((320,320))
        os.remove(os.path.join(DIRECTORY, file))
        img.save(os.path.join(DIRECTORY, file))
    

if __name__ == '__main__':
    resize_images("/Volumes/90OL67YGN/images/Edema")