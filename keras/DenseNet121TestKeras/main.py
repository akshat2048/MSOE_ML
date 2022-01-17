import keras
from keras import layers
import pandas as pd
import numpy as np

import environmentsettings

def create_data_set():
    # Create a dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(environmentsettings.settings['TRAINING_DIRECTORY'], batch_size=environmentsettings.settings['BATCH_SIZE'], image_size=(256, 256))

    '''
    image_size: Size to resize images to after they are read from disk. Defaults to (256, 256). Since the pipeline processes batches of images that must all have the same size, this must be provided.

    ^^This is from the docs
    I assume that this means that the images are resized to the image size provided
    '''

    # For demonstration, iterate over the batches yielded by the dataset.
    for data, labels in train_dataset:
        print(data.shape)  # (64, 200, 200, 3)
        print(data.dtype)  # float32
        print(labels.shape)  # (64,)
        print(labels.dtype)  # int32

    return train_dataset





