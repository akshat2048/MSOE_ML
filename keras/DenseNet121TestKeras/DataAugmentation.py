
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
import tensorflow.keras.layers as layers

import tensorflow.data.AUTOTUNE as AUTOTUNE


import environmentsettings

def main():
    print("Starting")
    train_dataset = augment_datasets()
    valid_dataset = create_validation_data_set()
    model = create_model()

    model.compile(
        optimizer=keras.optimizers.Adam(lr=environmentsettings.settings['LEARNING_RATE']),
        loss='binary_crossentropy',
        metrics=['Accuracy']
    )

    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=environmentsettings.settings['EPOCHS']
    )

    

    print(history.history)

def augment_datasets():
    data_augmentation = keras.Sequential(
        layers.experimental.preprocessing.RandomFlip('horizontal'),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomContrast(0.1),
        layers.experimental.preprocessing.RandomTranslation(0.1),
    )
    
    augmented_dataset = create_training_data_set(print_dataset=False).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    return augmented_dataset.prefetch(buffer_size=AUTOTUNE)

def create_training_data_set(print_dataset=False):
    # Create a dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        environmentsettings.settings['TRAINING_DIRECTORY'], 
        batch_size=environmentsettings.settings['BATCH_SIZE'], 
        image_size=(224, 224), 
        color_mode='rgb',
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
        include_top=True,
        pooling='avg',
        weights=None,
    )
    
    return model

def create_validation_data_set(print_dataset=False):
    # Create a dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        environmentsettings.settings['TESTING_DIRECTORY'], 
        batch_size=environmentsettings.settings['BATCH_SIZE'], 
        image_size=(224, 224), 
        color_mode='rgb',
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