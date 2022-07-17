
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras_adabound import AdaBound
import environmentsettings
from tensorflow.keras.applications import DenseNet121

# Set up the dataframe
# I just picked two random diseases and used them as normal and abnormal

NAME_OF_TRAIN_FOLDER_CONTAINING_NORMAL_CLASSES = '/Volumes/90OL67YGN/images/train/Pneumonia'
NAME_OF_TRAIN_FOLDER_CONTAINING_ABNORMAL_CLASSES = '/Volumes/90OL67YGN/images/train/Pneumothorax'

lst_of_files_and_classes = []

for root, dirs, files in os.walk('/Volumes/90OL67YGN/images/train'):
    for file in files:
        if (root == NAME_OF_TRAIN_FOLDER_CONTAINING_NORMAL_CLASSES):
            lst_of_files_and_classes.append((os.path.join(root, file), "Normal"))
        elif (root == NAME_OF_TRAIN_FOLDER_CONTAINING_ABNORMAL_CLASSES):
            lst_of_files_and_classes.append((os.path.join(root, file), "Abnormal"))

train_data = pd.DataFrame(lst_of_files_and_classes, columns =['filename', 'label'])
train_data = train_data.sample(frac=1).reset_index(drop=True)

# Set up the actual k-fold stuff

Y = train_data[['label']]

kf = KFold(n_splits = 5)

idg = ImageDataGenerator()

fold_var = 1
VALIDATION_ACCURACY = []
VALIDATION_LOSS = []

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

    outputs = keras.layers.Dense(1, activation = 'sigmoid')(x)



    # Combine inputs and outputs to create model

    model = keras.Model(inputs, outputs)



    # uncomment the following code if you want to see the model

    return model


for train_index, val_index in kf.split(np.zeros(len(Y)),Y):
    training_data = train_data.iloc[train_index]
    validation_data = train_data.iloc[val_index]
    train_data_generator = idg.flow_from_dataframe(training_data, x_col = "filename", y_col = "label", class_mode = "binary", shuffle = True)
    valid_data_generator  = idg.flow_from_dataframe(validation_data, x_col = "filename", y_col = "label", class_mode = "binary", shuffle = True)
	
	# CREATE NEW MODEL
    model = create_model()
	# COMPILE NEW MODEL
    model.compile(

        optimizer= AdaBound(name = 'AdaBound', learning_rate = 0.00001),

        loss='binary_crossentropy',

        metrics=['accuracy']

    )
	
	# CREATE CALLBACKS
    checkpoint = keras.callbacks.ModelCheckpoint('C:/Users/samee/Documents/Imagine Cup Saved Models/AP Fine Tuned/{fold_var}{epoch:02d}-{val_loss:.2f}.h5',

        monitor = 'val_loss',

        mode = 'min'

    )

    callbacks_list = [checkpoint]
	# There can be other callbacks, but just showing one because it involves the model name
	# This saves the best model
	# FIT THE MODEL
    history = model.fit(train_data_generator,
			    epochs=environmentsettings.setting_binary['EPOCHS'],
			    callbacks=callbacks_list,
			    validation_data=valid_data_generator)
	#PLOT HISTORY
	#		:
	#		:
    print(history.history)
    
	# LOAD BEST MODEL to evaluate the performance of the model
    # model.load_weights("/saved_models/model_"+str(fold_var)+".h5")
	
    results = model.evaluate(valid_data_generator)
    results = dict(zip(model.metrics_names,results))
	
    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])
	
    tf.keras.backend.clear_session()
	
    fold_var += 1