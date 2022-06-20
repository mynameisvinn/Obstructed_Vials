import numpy as np
from tensorflow import keras
import tensorflow as tf

INP_SIZE = (200, 200)


def check_for_obstructions(image: np.ndarray) -> bool:
    """Returns True if there are obstructions in the image.
    """

    MODEL_PATH = "./model_weights"
    model = keras.models.load_model(MODEL_PATH)

    image = preprocess_dataset(image)
    predictions = model.predict(image)
    return np.argmax(predictions, axis=1).astype(bool)[0]


def preprocess_dataset(image: np.ndarray):
    """Preprocess image for inference.
    Args:
        image (np.ndarray): 
    Returns:
        image: tensorflow.python.framework.ops.EagerTensor
    """
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize(image, (INP_SIZE[0], INP_SIZE[1]))
    return image