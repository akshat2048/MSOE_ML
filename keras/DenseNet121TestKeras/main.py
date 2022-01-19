from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np

import environmentsettings

def create_training_data_set(print_dataset=False):
    # Create a dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        environmentsettings.settings['TRAINING_DIRECTORY'], 
        batch_size=environmentsettings.settings['BATCH_SIZE'], 
        image_size=(256, 256), 
        color_mode='grayscale', 
        crop_to_aspect_ratio=True,
        label_mode='binary'
    )

    '''
    image_size: Size to resize images to after they are read from disk. Defaults to (256, 256). Since the pipeline processes batches of images that must all have the same size, this must be provided.

    ^^This is from the docs
    I assume that this means that the images are resized to the image size provided
    '''

    '''
    label_mode: - 'int': means that the labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss). - 'categorical' means that the labels are encoded as a categorical vector (e.g. for categorical_crossentropy loss). - 'binary' means that the labels (there can be only 2) are encoded as float32 scalars with values 0 or 1 (e.g. for binary_crossentropy). - None (no labels).
    '''

    # For demonstration, iterate over the batches yielded by the dataset.
    if not print_dataset:
        return train_dataset
    
    for data, labels in train_dataset:
        print(data.shape)
        print(data.dtype)
        print(labels.shape) 
        print(labels.dtype)

    return train_dataset

def create_validation_data_set(print_dataset=False):
    # Create a dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        environmentsettings.settings['TESTING_DIRECTORY'], 
        batch_size=environmentsettings.settings['BATCH_SIZE'], 
        image_size=(256, 256), 
        color_mode='grayscale', 
        crop_to_aspect_ratio=True,
        label_mode='binary'
    )

    '''
    image_size: Size to resize images to after they are read from disk. Defaults to (256, 256). Since the pipeline processes batches of images that must all have the same size, this must be provided.

    ^^This is from the docs
    I assume that this means that the images are resized to the image size provided
    '''

    '''
    label_mode: - 'int': means that the labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss). - 'categorical' means that the labels are encoded as a categorical vector (e.g. for categorical_crossentropy loss). - 'binary' means that the labels (there can be only 2) are encoded as float32 scalars with values 0 or 1 (e.g. for binary_crossentropy). - None (no labels).
    '''

    # For demonstration, iterate over the batches yielded by the dataset.
    if not print_dataset:
        return train_dataset
    
    for data, labels in train_dataset:
        print(data.shape)
        print(data.dtype)
        print(labels.shape) 
        print(labels.dtype)

    return train_dataset

def create_model():
    model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(256, 256, 1),
        pooling='avg'
    )
    
    return model

def main():
    train_dataset = create_training_data_set()
    model = create_model()

    model.compile(
        optimizer=keras.optimizers.Adam(lr=environmentsettings.settings['LEARNING_RATE']),
        loss=keras.losses.binary_crossentropy(from_logits=True),
        metrics=[keras.metrics.AUC(from_logits=True), keras.metrics.BinaryAccuracy()]
    )

    history = model.fit(
        train_dataset,
        epochs=environmentsettings.settings['EPOCHS']
    )

    print(history.history)
    # preprocess the data
    # https://keras.io/preprocessing/image/


main()





