from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
import tensorflow.keras.layers as layers
import numpy as np
import environmentsettings

def create_training_data_set(print_dataset=False):
    # Create a dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        environmentsettings.setting_categorical['TRAINING_DIRECTORY'], 
        batch_size=environmentsettings.setting_categorical['BATCH_SIZE'], 
        image_size=(224, 224), 
        color_mode='rgb',
        label_mode='categorical'
    )

    '''
    image_size: Size to resize images to after they are read from disk. Defaults to (256, 256). Since the pipeline processes batches of images that must all have the same size, this must be provided.

    ^^This is from the docs
    I assume that this means that the images are resized to the image size provided
    '''

    '''
    label_mode: - 'int': means that the labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss). - 'categorical' means that the labels are encoded as a categorical vector (e.g. for categorical_crossentropy loss). - 'binary' means that the labels (there can be only 2) are encoded as float32 scalars with values 0 or 1 (e.g. for binary_crossentropy). - None (no labels).
    '''
    return train_dataset

def create_validation_data_set(print_dataset=False):
    # Create a dataset
    test_dataset = keras.preprocessing.image_dataset_from_directory(
        environmentsettings.setting_categorical['TESTING_DIRECTORY'], 
        batch_size=environmentsettings.setting_categorical['BATCH_SIZE'], 
        image_size=(224, 224), 
        color_mode='rgb',
        label_mode='categorical'
    )

    return test_dataset

def create_model():
    model = DenseNet121(
        include_top=True,
        weights=None,
        classes=1
    )
    
    return model

def main():
    print("Starting")
    train_dataset = create_training_data_set()
    # print(np.concatenate([y for x, y in train_dataset], axis = 0).shape)
    model = create_model()
    # model.layers[-1].activation = keras.activations.sigmoid
    # model = keras.utils.apply_modifications(model)

    model.compile(
        optimizer= keras.optimizers.SGD(learning_rate=environmentsettings.setting_categorical['LEARNING_RATE']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    # optimizer = keras.optimizers.Adam(lr=environmentsettings.setting_categorical['LEARNING_RATE'])
    # TRY ADAM WHEN YOU GET HOME, USING SGD RIGHT NOW
    history = model.fit(
        train_dataset,
        epochs=environmentsettings.setting_categorical['EPOCHS']
    )

    print(history.history)
    # preprocess the data
    # https://keras.io/preprocessing/image/


if __name__ == "__main__":
    main()