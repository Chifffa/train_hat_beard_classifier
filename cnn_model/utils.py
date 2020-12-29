from typing import Optional, List, Any

import keras
import numpy as np
from sklearn.metrics import classification_report

from .custom_models import CustomResNet18
from config import MODEL_TYPE, INPUT_SHAPE, INPUT_NAME, NUM_CLASSES, ALPHA, ACTIVATION_TYPE, REGULARIZATION


def get_model(weights: Optional[str]) -> keras.models.Model:
    """
    Creating keras.model.Model object with pretrained or loaded from file weights.

    :param weights: path to saved model weights.
    :return: object keras.model.Model().
    """
    if MODEL_TYPE.lower() == 'custom_resnet18':
        model_class = CustomResNet18(INPUT_SHAPE, NUM_CLASSES, ALPHA, REGULARIZATION, ACTIVATION_TYPE, INPUT_NAME)
        model = model_class.build()
    else:
        raise TypeError('Wrong MODEL_TYPE: can be only "custom_resnet18" now, got "{}".'.format(MODEL_TYPE))
    if weights is not None:
        model.load_weights(weights)
    return model


def calculate_metrics(labels: List[Any], predicts: List[Any], label_names: Optional[List[str]] = None) -> str:
    """
    Counting metrics during validation.

    :param labels: list with labels from data generator.
    :param predicts: list with predictions from model object (KerasModelInference, TFGraphLoader or OnnxModelLoader).
    :param label_names: list with class names.
    """
    all_labels = np.concatenate(labels, axis=0)
    all_predicts = np.concatenate(predicts, axis=0)
    report = classification_report(
        np.argmax(all_labels, axis=-1),
        np.argmax(all_predicts, axis=-1),
        target_names=label_names
    )
    return report


class KerasModelInference:
    def __init__(self, weights: Optional[str] = None) -> None:
        """
        A wrapper over keras.model.predict for easy use with TFGraphLoader and OnnxModelLoader.

        :param weights: path to saved keras model weights.
        """
        self.model = get_model(weights)

    def inference(self, image: np.ndarray) -> List[Any]:
        """
        Run inference on single image.

        :param image: image numpy array with shape (batch_size, height, width, channels).
        :return: list with outputs.
        """
        return self.model.predict(image)
