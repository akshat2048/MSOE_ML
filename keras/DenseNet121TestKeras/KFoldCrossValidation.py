
from platform import mac_ver
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
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics

# Set up the dataframe
# I just picked two random diseases and used them as normal and abnormal

NAME_OF_TRAIN_FOLDER_CONTAINING_NORMAL_CLASSES = 'C:\\Users\\samee\\Downloads\\Fracture+Normal\\Normal'
NAME_OF_TRAIN_FOLDER_CONTAINING_ABNORMAL_CLASSES = 'C:\\Users\\samee\\Downloads\\Fracture+Normal\\Fracture'

lst_of_files_and_classes = []

for root, dirs, files in os.walk('C:\\Users\\samee\\Downloads\\Fracture+Normal'):
    for file in files:
        if (root == NAME_OF_TRAIN_FOLDER_CONTAINING_NORMAL_CLASSES):
            lst_of_files_and_classes.append((os.path.join(root, file), "Normal", 1.0))
        elif (root == NAME_OF_TRAIN_FOLDER_CONTAINING_ABNORMAL_CLASSES):
            lst_of_files_and_classes.append((os.path.join(root, file), "Fracture", 0.0))

train_data = pd.DataFrame(lst_of_files_and_classes, columns =['filename', 'label', 'label2'])
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

def plot_and_save_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues,
                        save_path=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(save_path)


def find_cm(y_pred, test_dataset):
    labels = np.array([])
    for x in test_dataset['label2']:
        labels = np.append(labels, [x], axis = 0)
    prediction = np.round(y_pred)

    cm = metrics.confusion_matrix(labels, prediction)
    return cm

for train_index, val_index in kf.split(np.zeros(len(Y)),Y):
    
    training_data = train_data.iloc[train_index]
    validation_data = train_data.iloc[val_index]
    train_data_generator = idg.flow_from_dataframe(training_data, x_col = "filename", y_col = "label", class_mode = "binary", shuffle = True)
    valid_data_generator  = idg.flow_from_dataframe(validation_data, x_col = "filename", y_col = "label", class_mode = "binary", shuffle = False)


	# CREATE NEW MODEL
    model = create_model()
	# COMPILE NEW MODEL
    model.compile(

        optimizer= AdaBound(name = 'AdaBound', learning_rate = 0.00001),

        loss='binary_crossentropy',

        metrics=['accuracy']

    )
    direc = f'C:/Users/samee/Documents/Imagine Cup Saved Models/K-Fold Cross Validation/{fold_var}'

    if not os.path.exists(direc):
        os.makedirs(direc)
	
	# CREATE CALLBACKS
    checkpoint = keras.callbacks.ModelCheckpoint(f'C:/Users/samee/Documents/Imagine Cup Saved Models/K-Fold Cross Validation/{fold_var}' + '/{epoch:02d}-{val_accuracy:.4f}.h5',

        monitor = 'val_loss',

        mode = 'min'

    )

    callbacks_list = [checkpoint]
	# There can be other callbacks, but just showing one because it involves the model name
	# This saves the best model
	# FIT THE MODEL
    history = model.fit(train_data_generator,
			    epochs= 10,
			    callbacks=callbacks_list,
			    validation_data=valid_data_generator)
	#PLOT HISTORY
	#		:
	#		:
    print(history.history)

    val = max(history.history['val_accuracy'])
    ep = history.history['val_accuracy'].index(val) + 1
        
    
	# LOAD BEST MODEL to evaluate the performance of the model
    model.load_weights(f'{direc}/0{ep}-{val:.4f}.h5')
    
	
    results = model.predict(valid_data_generator)
    # results = dict(zip(model.metrics_names,results))

    cm = find_cm(results, validation_data)
    plot_and_save_confusion_matrix(cm, classes = ['Fracture', 'Normal'], save_path = f'C:/Users/samee/Documents/Imagine Cup Saved Models/K-Fold Cross Validation/{fold_var}/confusion_matrix {fold_var}.png')
    print(cm)
	
    # VALIDATION_ACCURACY.append(results['accuracy'])
    # VALIDATION_LOSS.append(results['loss'])
	
    tf.keras.backend.clear_session()
    plt.clf()
	
    fold_var += 1
