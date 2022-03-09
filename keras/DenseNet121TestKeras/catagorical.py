from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Reshape, GlobalAveragePooling2D
from tensorflow.keras import Sequential
import numpy as np
import environmentsettings
from keras.callbacks import Callback



def create_training_data_set(print_dataset=False):
    # Create a dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        environmentsettings.setting_categorical['TRAINING_DIRECTORY'], 
        batch_size=environmentsettings.setting_categorical['BATCH_SIZE'], 
        image_size=(224, 224), 
        color_mode='rgb',
        label_mode='categorical',
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
    # This is for a new model
    model = create_model()
    model.layers[-1].activation = keras.activations.softmax
    # This is for a saved model
    # model = keras.models.load_model('C:/Users/samee/Documents/Imagine Cup Saved Models/NIH Categorical Save')
    # earlyStopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min')
    #Put your own file path in but keep the {epoch:02d} and onwards. Its to save after every epoch
    checkpoint = keras.callbacks.ModelCheckpoint('C:/Users/samee/Documents/Imagine Cup Saved Models/NIH Categorical Callbacks/{epoch:02d}-{val_loss:.2f}.h5',
        monitor = 'val_loss',
        mode = 'min'
    )

    model.compile(
        optimizer= keras.optimizers.SGD(learning_rate=environmentsettings.setting_categorical['LEARNING_RATE'], momentum = 0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # keras.optimizers.Adam(lr=environmentsettings.setting_categorical['LEARNING_RATE'])
    history = model.fit(
        train_dataset,
        epochs=environmentsettings.setting_categorical['EPOCHS'],
        validation_data = create_validation_data_set(),
        callbacks = [checkpoint]
    )

    model.save('C:/Users/samee/Documents/Imagine Cup Saved Models/NIH Categorical Save Correct')

    print(history.history)
    # preprocess the data
    # https://keras.io/preprocessing/image/


if __name__ == "__main__":
    main()