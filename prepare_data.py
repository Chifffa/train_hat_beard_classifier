import os
import json
import argparse
from typing import List, Tuple, Dict, Union

import numpy as np
from cv_utils import init_random_generators

from config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, NO_HAT_BEARD_PATH, HAT_PATH, BEARD_PATH, HAT_BEARD_PATH, TRAIN_TEST_SPLIT,
    DATA_PATH
)
from cnn_model import test_data_generator


def _get_relative_paths(dataset_path: str, folder_name: str, is_hat: bool,
                        is_beard: bool) -> Tuple[List[Dict[str, str]], List[Dict[str, Union[str, bool]]]]:
    """
    Get relative paths to images for train and test modes.

    :param dataset_path: path to full dataset.
    :param folder_name: current folder name (contains images of single class).
    :param is_hat: True if images with hat.
    :param is_beard: True if images with beard.
    :return: train and test data.
    """
    images_list = [os.path.join(folder_name, p) for p in os.listdir(os.path.join(dataset_path, folder_name))]
    np.random.shuffle(images_list)

    test_data = [
        {'path': p, 'hat': is_hat, 'beard': is_beard} for p in images_list[:int(TRAIN_TEST_SPLIT * len(images_list))]
    ]
    train_data = [
        {'path': p, 'hat': is_hat, 'beard': is_beard} for p in images_list[int(TRAIN_TEST_SPLIT * len(images_list)):]
    ]
    return train_data, test_data


def prepare_dataset(dataset_path: str) -> None:
    """
    Creating a dataset for hat/beard classification.

    :param dataset_path: path to full dataset.
    """
    no_h_b_train, no_h_b_test = _get_relative_paths(dataset_path, NO_HAT_BEARD_PATH, is_hat=False, is_beard=False)
    h_train, h_test = _get_relative_paths(dataset_path, HAT_PATH, is_hat=True, is_beard=False)
    b_train, b_test = _get_relative_paths(dataset_path, BEARD_PATH, is_hat=False, is_beard=True)
    h_b_train, h_b_test = _get_relative_paths(dataset_path, HAT_BEARD_PATH, is_hat=True, is_beard=True)

    os.makedirs(DATA_PATH, exist_ok=True)
    with open(TRAIN_DATA_PATH, 'w') as file:
        json.dump(no_h_b_train + h_train + b_train + h_b_train, file, indent=4)
    with open(TEST_DATA_PATH, 'w') as file:
        json.dump(no_h_b_test + h_test + b_test + h_b_test, file, indent=4)


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('Creating a dataset for hat/beard classification.')
    parser.add_argument('dataset_path', type=str, help='Path to dataset folder.')
    parser.add_argument('--test', action='store_true', help='Test data generator.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random generators seed. If seed < 0, seed will be None.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.seed >= 0:
        init_random_generators(args.seed)
    prepare_dataset(args.dataset_path)
    if args.test:
        test_data_generator(args.dataset_path, True)
