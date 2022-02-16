import imghdr
import os

def check_images(DIRECTORY, DELETE_NON_IMAGES=False):
        
    for file in os.listdir(DIRECTORY):
        if imghdr.what(os.path.join(DIRECTORY, file)) != 'png':
            print(os.path.join(DIRECTORY, file))
            if DELETE_NON_IMAGES:
                os.remove(os.path.join(DIRECTORY, file))
    
    if DELETE_NON_IMAGES:
        for file in os.listdir(DIRECTORY):
            if imghdr.what(os.path.join(DIRECTORY, file)) != 'png':
                print(os.path.join(DIRECTORY, file))