from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
import tensorflow.keras.layers as layers

import environmentsettings


def create_training_data_set(print_dataset=False):
    # Create a dataset
    # Use CSV to select certain type of condition
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
    model = create_model()
    model.layers[-1].activation = keras.activations.sigmoid
    # model = keras.utils.apply_modifications(model)

    model.compile(
        optimizer= keras.optimizers.SGD(learning_rate=environmentsettings.setting_binary['LEARNING_RATE']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    # optimizer= keras.optimizers.SGD(learning_rate=environmentsettings.setting_binary['LEARNING_RATE'])
    # TRY ADAM WHEN YOU GET HOME, USING SGD RIGHT NOW
    history = model.fit(
        train_dataset,
        epochs=environmentsettings.setting_binary['EPOCHS'],
        validation_data = create_validation_data_set()
        # batch_size = environmentsettings.setting_binary['BATCH_SIZE']
    )

    print(history.history)
    # preprocess the data
    # https://keras.io/preprocessing/image/


if __name__ == "__main__":
    main()
