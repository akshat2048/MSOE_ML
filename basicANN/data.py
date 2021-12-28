# import pandas and numpy
import numpy as np
import pandas as pd
import shutil
import os

def getDataframe():
    # read in csv
    data = pd.read_csv('datasetsforeverything/chest_x_ray_images_labels_sample.csv')

    # get the imageIDs of images with a projection value of 'PA'
    images = data[data['Projection'] == 'PA']

    # #get a list of those ImageIDs
    # imageIDs = images['ImageID'].tolist()

    images['Classification'] = images['Labels'].str.contains('normal')

    images.index = images['ImageID']

    # get a list of images with a classification of 'normal'
    normal_images = images[images['Classification'] == True]
    # print(normal_images.Classification)

    abnormal_images = images[images['Classification'] == False]
    # print(abnormal_images.Classification)

    for image in normal_images['ImageID']:
        sourcepath = os.path.join('datasetsforeverything/sample', image)
        destpath = 'datasetsforeverything/sample/normal/'
        shutil.move(sourcepath, destpath)
    
    for image in abnormal_images['ImageID']:
        sourcepath = os.path.join('datasetsforeverything/sample', image)
        destpath = 'datasetsforeverything/sample/abnormal'
        shutil.move(sourcepath, destpath)
    
if __name__ == '__main__':
    getDataframe()