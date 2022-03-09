from venv import create
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Reshape, GlobalAveragePooling2D
from tensorflow.keras import Sequential
import numpy as np
import environmentsettings
from keras.callbacks import Callback
import keras.backend as K
import matplotlib.pyplot as plt
import math

class LRFinder(Callback):
    
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset. 
    
    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5, 
                                 max_lr=1e-2, 
                                 steps_per_epoch=np.ceil(epoch_size/batch_size), 
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])
            
            lr_finder.plot_loss()
        ```
    
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
        
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''
    
    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}
        
    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations 
        return self.min_lr + (self.max_lr-self.min_lr) * x
        
    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)
        
    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        K.set_value(self.model.optimizer.lr, self.clr())
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.show()
        
    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()


def create_training_data_set(print_dataset=False):
    # Create a dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        environmentsettings.setting_binary['TRAINING_DIRECTORY'], 
        batch_size=environmentsettings.setting_binary['BATCH_SIZE'], 
        image_size=(224, 224), 
        color_mode='rgb',
        label_mode='binary'
    )

    return train_dataset

def create_validation_data_set(print_dataset=False):
    # Create a dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        environmentsettings.setting_binary['TESTING_DIRECTORY'], 
        batch_size=environmentsettings.setting_binary['BATCH_SIZE'],
        image_size=(224, 224), 
        color_mode='rgb',
        label_mode='binary'
    )


    return train_dataset

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

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

def main():
    print("Starting")
    train_dataset = create_training_data_set()
    valid_dataset = create_validation_data_set()
    # For a new model
    model = create_model()
    
    model.compile(
        optimizer= keras.optimizers.SGD(learning_rate=environmentsettings.setting_binary['LEARNING_RATE'], momentum = 0.9),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    # keras.optimizers.SGD(learning_rate=environmentsettings.setting_binary['LEARNING_RATE'], momentum = 0.9)
    # keras.optimizers.Adam(learning_rate=environmentsettings.setting_binary['LEARNING_RATE'])
    #Try adam to see if it work on this batchsize

    #After you train the two initial weights
    # model = after_two_weights()

    checkpoint = keras.callbacks.ModelCheckpoint('C:/Users/samee/Documents/Imagine Cup Saved Models/Combined Binary Callbacks/{epoch:02d}-{val_loss:.2f}.h5',
        monitor = 'val_loss',
        mode = 'min'
    )

    lr_finder = LRFinder(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=len(create_training_data_set()), epochs = environmentsettings.setting_binary['EPOCHS'])
    
    history = model.fit(
        train_dataset,
        epochs=environmentsettings.setting_binary['EPOCHS'],
        validation_data=valid_dataset,
        callbacks = [lr_finder]
    )

    lr_finder.plot_loss()
    lr_finder.plot_lr()

    print(history.history)
    # preprocess the data
    # https://keras.io/preprocessing/image/

def after_two_weights():
    model = keras.models.load_model('C:/Users/samee/Documents/Imagine Cup Saved Models/after two weights.h5')
    # model.trainable = True

    return model

if __name__ == "__main__":
    # model = after_two_weights()
    # print("weights:", len(model.weights))
    # print("trainable_weights:", len(model.trainable_weights))
    # print("non_trainable_weights:", len(model.non_trainable_weights))
    # print(model.summary())
    main()




