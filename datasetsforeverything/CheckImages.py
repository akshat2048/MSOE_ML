import imghdr
import os

mounted_path = '/media/pi/'

for directory in os.listdir(mounted_path):
    data_dir = os.path.join(mounted_path, directory)
    
    for file in os.listdir(data_dir):
        if imghdr.what(os.path.join(data_dir, file)) != 'png':
            print(os.path.join(data_dir, file))
            os.remove(os.path.join(data_dir, file))

    for file in os.listdir(data_dir):
        if imghdr.what(os.path.join(data_dir, file)) != 'png':
            print(os.path.join(data_dir, file))