from configparser import LegacyInterpolation
from venv import create
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Reshape, GlobalAveragePooling2D
from tensorflow.keras import Sequential
import numpy as np
import environmentsettings
from keras.callbacks import Callback
import keras.backend as K
import matplotlib.pyplot as plt
import math
from keras_adabound import AdaBound

def create_training_data_set():
    # Create a dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        environmentsettings.setting_binary['TRAINING_DIRECTORY'], 
        batch_size=environmentsettings.setting_binary['BATCH_SIZE'], 
        image_size=(224, 224), 
        color_mode='rgb',
        label_mode='binary'
    )

    return train_dataset

def create_validation_data_set():
    # Create a dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        environmentsettings.setting_binary['TESTING_DIRECTORY'], 
        batch_size= 32,
        image_size=(224, 224), 
        color_mode='rgb',
        label_mode='binary'
    )


    return train_dataset

def create_model():
    base_model = MobileNetV3Small(
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

    # Add a batch normalization layer
    x = keras.layers.BatchNormalization()(x)

    # Add final dense layer with 6 classes for the 6 types of fruit
    outputs = keras.layers.Dense(1, activation = 'sigmoid')(x)

    # Combine inputs and outputs to create model
    model = keras.Model(inputs, outputs)

    # uncomment the following code if you want to see the model
    return model

def main():
    print("Starting")
    train_dataset = create_training_data_set()
    valid_dataset = create_validation_data_set()
    # For a new model
    model = create_model()

    # model.trainable = True

    # model.load_weights('C:/Users/samee/Documents/Imagine Cup Saved Models/Binary/AP with Adabound mobile 2.0/05-0.42.h5')
    
    # model.compile(
    #     optimizer= AdaBound(name = 'AdaBound', learning_rate = 0.0001),
    #     loss='binary_crossentropy',
    #     metrics=['accuracy']
    # )
    # keras.optimizers.SGD(learning_rate=environmentsettings.setting_binary['LEARNING_RATE'], momentum = 0.9)
    # keras.optimizers.Adam(learning_rate=environmentsettings.setting_binary['LEARNING_RATE'])
    # keras.optimizers.Adamax(learning_rate = environmentsettings.setting_binary['LEARNING_RATE'])

    #After you train the two initial weights
    # model = after_two_weights()

    model.compile(
        optimizer= AdaBound(name = 'AdaBound', learning_rate = 0.0001, final_lr= 0.01),
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )

    # Use SGD for fine tuning
    # model.compile(
    #     optimizer= keras.optimizers.SGD(learning_rate=environmentsettings.setting_binary['LEARNING_RATE'], momentum = 0.9),
    #     loss='binary_crossentropy',
    #     metrics=['accuracy']
    # )

    checkpoint = keras.callbacks.ModelCheckpoint('C:/Users/samee/Documents/Imagine Cup Saved Models/Binary/PneumoniaUnder/{epoch:02d}-{val_loss:.2f}.h5',
        monitor = 'val_loss',
        mode = 'min'
    )

    # sched = keras.callbacks.LearningRateScheduler(scheduler)

    # lr_finder = LRFinder(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=len(create_training_data_set()), epochs = environmentsettings.setting_binary['EPOCHS'])
    
    history = model.fit(
        train_dataset,
        epochs=environmentsettings.setting_binary['EPOCHS'],
        validation_data=valid_dataset,
        callbacks = [checkpoint]
    )


    print(history.history)
    # preprocess the data
    # https://keras.io/preprocessing/image/
    # np.save('C:/Users/samee/Documents/Imagine Cup Saved Models/Graph-Data/PAsat/Pneumonia/Pneumonia.npy', history.history)


# Fine Tuning
def after_two_weights():
    model = create_model()
    model.load_weights('C:/Users/samee/Documents/Imagine Cup Saved Models/AP with Adabound/08-0.42.h5')
    model.trainable = True
    
    

    return model


# Random Model Method not important
def layer_trainable():
    model = create_model()
    model.load_weights('C:/Users/samee/Documents/Imagine Cup Saved Models/AP with Adabound no support/13-0.49.h5')
    print(model.summary())
    a = False
    for layer in model.layers[1].layers:
        if layer.name == 'conv2_block1_0_bn':
            a = True
        if a:
            layer.trainable = True
    return model

if __name__ == "__main__":
    main()
    
