from PIL import Image
import numpy as np
import os
from data import getDataframe
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def packageImages():
    df = getDataframe()
    images = []

    #for i in range(len(df)):
    # i = 0
    # path = os.path.join('datasetsforeverything/sample', df.iloc[i]['ImageID'])
    # image = Image.open(path)
    # image.show()
    # print(path)
    # convert to grayscale
    # img = mpimg.imread(os.path.join('datasetsforeverything/sample/', df.iloc[i]['ImageID']))
    # imgplot = plt.imshow(img)
    # plt.gray()
    # plt.show()
    # show the image size
    # print("Image Size of", os.path.join('datasetsforeverything/sample/', df.iloc[i]['ImageID']), ' ->', image.size)
    
    # save images
    # print(images[0])
    # np.save('datasetsforeverything/sample/images.npy', images)

    im = Image.open(os.path.join('datasetsforeverything/sample', df.iloc[0]['ImageID']))

    # PIL complains if you don't load explicitly
    im.load()

    # Get the alpha band
    alpha = im.split()[-1]

    im = im.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)

    # Set all pixel values below 128 to 255,
    # # and the rest to 0
    # mask = Image.eval(alpha, lambda a: 255 if a <=128 else 0)

    # Paste the color of index 255 and use alpha as a mask
    # im.paste(255, mask)

    # The transparency index is 255
    im.show()

if __name__ == '__main__':
    packageImages()

