import os
import shutil
import argparse
from typing import Optional

import keras
from cv_utils import get_date, freeze_keras_model, convert_frozen_graph_to_onnx, session_config

from cnn_model import get_model
from config import logger, WEIGHTS_PATH, INPUT_NAME, OUTPUT_NAMES


def convert(weights: Optional[str]) -> None:
    """
    Model conversion. The selected model will be saved in keras format (*.h5), tf 1.x frozen graph (*.pb)
    and onnx (*.onnx). Model will be saved in folder "WEIGHTS_PATH/_(current date)/".

    :param weights: path to saved keras model weights.
    """
    keras.backend.set_learning_phase(0)
    save_path = os.path.join(WEIGHTS_PATH, get_date()[2:])
    os.makedirs(save_path, exist_ok=True)
    if weights is None:
        logger.warning('\nNo weights provided. Converting random initialized model.\n')
        weights_name = 'random'
    else:
        weights_name = os.path.basename(weights)
        shutil.copyfile(weights, os.path.join(save_path, weights_name))
    model = get_model(weights)
    if weights is None:
        model.save(os.path.join(save_path, weights_name))
    frozen_model_path = freeze_keras_model(model, save_path, os.path.splitext(weights_name)[0], list(OUTPUT_NAMES))
    convert_frozen_graph_to_onnx(frozen_model_path, [INPUT_NAME], OUTPUT_NAMES, save_path, opset=None)


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('Script for converting trained models to frozen tf graph and onnx.')
    parser.add_argument('--weights', type=str, default=None, help='Path to saved model.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    session_config(None, set_keras=True)
    convert(args.weights)
