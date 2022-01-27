from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os

import environmentsettings

def fileNamesCSV():
    files = pd.read_csv('datasetsforeverything/NIH.csv')
    files['Classes'] = files['Finding Labels'].str.split('|')
    files = pd.DataFrame(files, columns=['Image Index', 'Classes'])
    files.rename(columns={'Image Index': 'filename', 'Classes' : 'labels'}, inplace=True)
    return files



def imageDataGen():
    img_gen = ImageDataGenerator(rescale=1/255, validation_split=0.2)
    img_metadata = fileNamesCSV()
    
    img_iter = img_gen.flow_from_dataframe(
        img_metadata,
        shuffle=True,
        directory=environmentsettings.settings['TRAINING_DIRECTORY'],
        x_col='filename',
        y_col='labels',
        class_mode='categorical',
        target_size=(128, 128),
        batch_size=20,
        subset='training'
    )

    img_iter_val = img_gen.flow_from_dataframe(
        img_metadata,
        shuffle=False,
        directory=environmentsettings.settings['TRAINING_DIRECTORY'],
        x_col='filename',
        y_col='labels',
        class_mode='categorical',
        target_size=(128, 128),
        batch_size=200,
        subset='validation'
    )

    return (img_iter, img_iter_val)

def create_model():
    model = DenseNet121(
        include_top=True,
        pooling='avg',
        weights=None,
    )
    
    return model

def main():
    model = create_model()

    model.compile(
        optimizer=keras.optimizers.Adam(lr=environmentsettings.settings['LEARNING_RATE']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit_generator(
        imageDataGen()[0],
        steps_per_epoch=100,
        epochs=10,
        validation_data=imageDataGen()[1],
        validation_steps=10
    )
    
    print(history.history)

if __name__ == '__main__':
    fileNamesCSV()