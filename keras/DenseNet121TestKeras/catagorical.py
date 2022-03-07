from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Reshape, GlobalAveragePooling2D
from tensorflow.keras import Sequential
import numpy as np
import environmentsettings
from sklearn.metrics import classification_report

def create_training_data_set(print_dataset=False):
    # Create a dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        environmentsettings.setting_categorical['TRAINING_DIRECTORY'], 
        batch_size=environmentsettings.setting_categorical['BATCH_SIZE'], 
        image_size=(224, 224), 
        color_mode='rgb',
        label_mode='categorical',
        validation_split=0.2,
        subset='training',
        seed=999
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
        environmentsettings.setting_categorical['TRAINING_DIRECTORY'], 
        batch_size=environmentsettings.setting_categorical['BATCH_SIZE'], 
        image_size=(224, 224), 
        color_mode='rgb',
        label_mode='categorical',
        validation_split=0.2,
        subset='validation',
        seed=999
    )

    return test_dataset

def create_model():

    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    # Next, we will freeze the base model so that all the learning from the ImageNet 
    # dataset does not get destroyed in the initial training.
    base_model.trainable = False

    # Create inputs with correct shape
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)

    # Add pooling layer or flatten layer
    x =  keras.layers.GlobalAveragePooling2D()(x)

    # Add final dense layer with 6 classes for the 6 types of fruit
    outputs = keras.layers.Dense(environmentsettings.setting_categorical['CLASSES'], activation = 'softmax')(x)

    # Combine inputs and outputs to create model
    model = keras.Model(inputs, outputs)

    # uncomment the following code if you want to see the model
    return model

def main():
    print("Starting")
    train_dataset = create_training_data_set()
    # print(np.concatenate([y for x, y in train_dataset], axis = 0).shape)
    model = create_model()
    # model = keras.utils.apply_modifications(model)

    model.compile(
        optimizer= keras.optimizers.Adam(learning_rate=environmentsettings.setting_categorical['LEARNING_RATE']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # optimizer = keras.optimizers.Adam(lr=environmentsettings.setting_categorical['LEARNING_RATE'])
    # TRY ADAM WHEN YOU GET HOME, USING SGD RIGHT NOW
    history = model.fit(
        train_dataset,
        epochs=environmentsettings.setting_categorical['EPOCHS'],
        validation_data = create_validation_data_set(),
        batch_size = environmentsettings.setting_categorical['BATCH_SIZE']
    )

    model.save(environmentsettings.setting_categorical['SAVE_DIRECTORY'] + '/NIH Categorical Save')

    print(history.history)
    # preprocess the data
    # https://keras.io/preprocessing/image/

def test():
    model = keras.models.load_model(environmentsettings.setting_categorical['SAVE_DIRECTORY'] + '/NIH Categorical Save')
    history = model.evaluate(
        create_validation_data_set()
    )
    print(classification_report)

if __name__ == "__main__":
    main()