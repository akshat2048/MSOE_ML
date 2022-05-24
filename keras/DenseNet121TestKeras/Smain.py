from tensorflow import keras

from tensorflow.keras.applications import DenseNet121

import environmentsettings

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



# Train the model

def main():

    print("Starting")

    train_dataset = create_training_data_set()

    valid_dataset = create_validation_data_set()



    # For a new model

    model = create_model()



    # If loading model then uncomment below line

    # model.load_weights('C:/Users/samee/Documents/Imagine Cup Saved Models/20-0.81.h5')



    #If fine tuning then uncomment below line

    # model = after_two_weights()



    model.compile(

        optimizer= AdaBound(name = 'AdaBound', learning_rate = 0.00001),

        loss='binary_crossentropy',

        metrics=['accuracy']

    )
    # keras.optimizers.SGD(learning_rate=environmentsettings.setting_binary['LEARNING_RATE'], momentum = 0.9)
    # keras.optimizers.Adam(learning_rate=environmentsettings.setting_binary['LEARNING_RATE'])
    # keras.optimizers.Adamax(learning_rate = environmentsettings.setting_binary['LEARNING_RATE'])



    # If need to use SGD for fine tuning

    # model.compile(

    #     optimizer= keras.optimizers.SGD(learning_rate=environmentsettings.setting_binary['LEARNING_RATE'], momentum = 0.9),

    #     loss='binary_crossentropy',

    #     metrics=['accuracy']

    # )



    # Check point callback that saves model after each iteration

    checkpoint = keras.callbacks.ModelCheckpoint('C:/Users/samee/Documents/Imagine Cup Saved Models/AP Fine Tuned/{epoch:02d}-{val_loss:.2f}.h5',

        monitor = 'val_loss',

        mode = 'min'

    )



    # Learning rate scheduler if need to be used with SGD

    # sched = keras.callbacks.LearningRateScheduler(scheduler)



    # Optimal learning rate finder 

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





# Fine Tuning

def after_two_weights():

    model = create_model()

    model.load_weights('C:/Users/samee/Documents/Imagine Cup Saved Models/AP with Adabound/08-0.42.h5')

    model.trainable = True  



    return model



if __name__ == "__main__":

    main()