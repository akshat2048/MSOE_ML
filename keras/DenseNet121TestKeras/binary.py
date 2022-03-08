from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
import environmentsettings

SAVE_NAME = 'AP32Adam_Binary'
OPTIMIZER = keras.optimizers.Adam(lr=environmentsettings.setting_categorical['LEARNING_RATE'])
AUGMENTING_DATA = False
CALLBACKS = []


def create_training_data_set():
    if not AUGMENTING_DATA:
        # Create a dataset
        train_dataset = keras.preprocessing.image_dataset_from_directory(
            environmentsettings.setting_categorical['TRAINING_DIRECTORY'], 
            batch_size=environmentsettings.setting_categorical['BATCH_SIZE'], 
            image_size=(224, 224), 
            color_mode='rgb',
            label_mode='binary',
            validation_split=0.2,
            subset='training',
            seed=999
        )
        return train_dataset
    else:
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

    if not AUGMENTING_DATA:
        # Create a dataset
        test_dataset = keras.preprocessing.image_dataset_from_directory(
            environmentsettings.setting_categorical['TRAINING_DIRECTORY'], 
            batch_size=environmentsettings.setting_categorical['BATCH_SIZE'], 
            image_size=(224, 224), 
            color_mode='rgb',
            label_mode='binary',
            validation_split=0.2,
            subset='validation',
            seed=999
        )

        return test_dataset
    else:
        test_dataset = keras.preprocessing.image_dataset_from_directory(
            environmentsettings.setting_binary['TESTING_DIRECTORY'], 
            batch_size=environmentsettings.setting_binary['BATCH_SIZE'], 
            image_size=(224, 224), 
            color_mode='rgb',
            label_mode='binary'
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
    outputs = keras.layers.Dense(1, activation = 'sigmoid')(x)

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
        optimizer= OPTIMIZER,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # optimizer = keras.optimizers.Adam(lr=environmentsettings.setting_categorical['LEARNING_RATE'])
    # TRY ADAM WHEN YOU GET HOME, USING SGD RIGHT NOW

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=environmentsettings.setting_categorical['SAVE_DIRECTORY'] + f'/{SAVE_NAME}_WEIGHTS',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    CALLBACKS.append(model_checkpoint_callback)

    history = model.fit(
        train_dataset,
        epochs=environmentsettings.setting_categorical['EPOCHS'],
        validation_data = create_validation_data_set(),
        batch_size=environmentsettings.setting_categorical['BATCH_SIZE'],
        callbacks=CALLBACKS
    )

    model.save(environmentsettings.setting_categorical['SAVE_DIRECTORY'] + f'/{SAVE_NAME}')

    print(history.history)
    # preprocess the data
    # https://keras.io/preprocessing/image/


if __name__ == "__main__":
    main()



