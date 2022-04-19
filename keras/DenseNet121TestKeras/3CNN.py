from tensorflow import keras

def create_model(models: list):
    """
    Generate a model that uses ensemble learning code to develop a 3-channel CNN

    Args:
        models (list): List of models to use, preferably loaded from .h5 files but could work with .pb as well
    """
    inputs = keras.Input(shape=(224, 224, 3))

    model_outputs = [model(inputs) for model in models]

    ensemble_output = keras.layers.Average()(model_outputs) # Simple averaging for ensemble learning, we can obviously make this more sophisticated if needed

    model = keras.Model(inputs=inputs, outputs=ensemble_output)

    return model

