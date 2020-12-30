import time
import argparse
from multiprocessing import cpu_count
from typing import Optional, List, Tuple

import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cv_utils import session_config, init_random_generators, TFGraphLoader, OnnxModelLoader
from cv_utils.classification import Recall, Precision, F1Score

from cnn_model import get_model, KerasModelInference, calculate_metrics, DataGenerator
from config import logger, LEARNING_RATE, NUM_CLASSES, INPUT_NAME, OUTPUT_NAMES


def evaluate(dataset_path: str, weights: Optional[str]) -> None:
    """
    Evaluate keras model on loss and training metrics. Function will print evaluating results on validation data.

    :param dataset_path: path to full dataset.
    :param weights: path to saved keras model weights.
    """
    train_data_gen = DataGenerator(dataset_path, is_train=True)
    test_data_gen = DataGenerator(dataset_path, is_train=False)
    model = get_model(weights)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  metrics=[Recall(NUM_CLASSES), Precision(NUM_CLASSES), F1Score(NUM_CLASSES)])
    message = 'Testing model "{}".\n\n'.format(weights)

    # Evaluate on training data.
    message += 'Train data:\n'
    results = model.evaluate_generator(train_data_gen, workers=min(24, cpu_count()), verbose=1)
    for name, res in zip(model.metrics_names, results):
        message += '{} = {:.04f}; '.format(name, res)
    message = message[:-2] + '\n\nTest data:\n'

    # Evaluate on testing data.
    results = model.evaluate_generator(test_data_gen, workers=min(24, cpu_count()), verbose=1)
    for name, res in zip(model.metrics_names, results):
        message += '{} = {:.04f}; '.format(name, res)
    message = message[:-2] + '\n'
    logger.info('\n' + message)


def predict(dataset_path: str, weights: Optional[str], show_predictions: bool, gpu_number: Optional[int]) -> None:
    """
    Validating model on test dataset, getting metrics and measuring inference time with batch size 1.
    Function will print validating results and measured time.

    :param dataset_path: path to full dataset.
    :param weights: path to saved model weights.
    :param show_predictions: if True, only show predictions, else get metrics and measure inference time.
    :param gpu_number: GPU index. If None then use CPU.
    """
    data_gen = DataGenerator(dataset_path, is_train=False, batch_size=1)

    if weights is None or weights.endswith('.h5'):
        model_obj = KerasModelInference(weights)
    elif weights.endswith('.pb'):
        model_obj = TFGraphLoader(weights, input_names=[INPUT_NAME], output_names=OUTPUT_NAMES, gpu_number=gpu_number)
    elif weights.endswith('.onnx'):
        # Onnx runtime is effective only on cpu.
        gpu_number = None
        model_obj = OnnxModelLoader(weights, on_cpu=True)
    else:
        msg = 'Weights "{}" have unknown format.'.format(weights)
        raise TypeError(msg)

    if show_predictions:
        for i in range(len(data_gen)):
            img_batch, label_batch = data_gen[i]
            predicts = model_obj.inference([img_batch])

            label, _, _ = get_class(label_batch)
            label = data_gen.classes[int(np.argmax(label))]

            prediction, hat_p, beard_p = get_class(predicts)
            prediction = data_gen.classes[int(np.argmax(prediction))]

            text = 'Label: "{}", predict: "{}",\nhat = {:.02f}%, beard = {:.02f}%'.format(
                label, prediction, hat_p * 100, beard_p * 100
            )
            plt.imshow(img_batch[0, :, :, :])
            plt.title(text)
            if plt.waitforbuttonpress(0):
                plt.close('all')
                return
            plt.close('all')
    else:
        images, all_times, all_predicts, all_labels = [], [], [], []
        for i in tqdm(range(len(data_gen)), desc='Reading images'):
            img, labels = data_gen[i]
            images.append(img)
            label_vector, _, _ = get_class(labels)
            all_labels.append([label_vector])
        for image in tqdm(images):
            start_time = time.time()
            predicts = model_obj.inference([image])
            finish_time = time.time()
            all_times.append(finish_time - start_time)
            predict_vector, _, _ = get_class(predicts)
            all_predicts.append([predict_vector])
        all_times = all_times[5:]
        message = '\nUsing model "{}". Using gpu: {}.\nMean inference time: {:.04f}. Mean FPS: {:.04f}.\n'.format(
            weights,
            gpu_number,
            np.mean(all_times),
            len(all_times) / sum(all_times)
        )
        message += calculate_metrics(all_labels, all_predicts, list(data_gen.classes))
        logger.info('\n' + message)


def get_class(classes_arrays: List[np.ndarray]) -> Tuple[np.ndarray, float, float]:
    """
    Get class in one-hot format for two outputs.

    :param classes_arrays: list of two arrays (hat one-hot and beard one-hot).
    :return: one-hot in 4 classes, corresponding to CLASS_NAMES and probabilities for 2 outputs.
    """
    hat_labels = classes_arrays[0][0]
    beard_labels = classes_arrays[1][0]

    hat_label = int(np.argmax(hat_labels))
    beard_label = int(np.argmax(beard_labels))
    if hat_label == 1 and beard_label == 1:
        return np.array([1, 0, 0, 0]), hat_labels[0], beard_labels[0]
    elif hat_label == 0 and beard_label == 1:
        return np.array([0, 1, 0, 0]), hat_labels[0], beard_labels[0]
    elif hat_label == 1 and beard_label == 0:
        return np.array([0, 0, 1, 0]), hat_labels[0], beard_labels[0]
    else:
        return np.array([0, 0, 0, 1]), hat_labels[0], beard_labels[0]


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('Script for evaluating, validating classifier and visualizing inference results.')
    parser.add_argument('mode', type=str, choices=['evaluate', 'validate', 'predict'],
                        help='Mode of working:\n'
                             '"evaluate" - evaluate keras model loss and metrics on validation data generator;\n'
                             '"validate" - validate model metrics and performance on validation data;\n'
                             '"predict" - show predictions on validation data. Mouse click to continue or press any'
                             ' button to exit.')
    parser.add_argument('dataset_path', type=str, help='Path to dataset folder.')
    parser.add_argument('-d', '--device', type=int, default=0, help='GPU device index.')
    parser.add_argument('--weights', type=str, default=None, help='Path to saved model.')
    parser.add_argument('--cpu', action='store_true', help='Use only cpu.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random generators seed. If seed < 0, seed will be None.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.seed >= 0:
        init_random_generators(args.seed)
    if args.cpu:
        gpu_id = None
        session_config(gpu_id, set_keras=True)
    else:
        gpu_id = args.device
        session_config([gpu_id], set_keras=True)
    keras.backend.set_learning_phase(0)
    if args.weights is None:
        logger.warning('No weights provided. Using random initialized model.')

    if args.mode == 'evaluate':
        evaluate(args.dataset_path, args.weights)
    elif args.mode == 'validate':
        predict(args.dataset_path, args.weights, show_predictions=False, gpu_number=gpu_id)
    elif args.mode == 'predict':
        predict(args.dataset_path, args.weights, show_predictions=True, gpu_number=gpu_id)
