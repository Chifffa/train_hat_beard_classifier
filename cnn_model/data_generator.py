import os
import json
from typing import Tuple, List

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv_utils.augmentation as aug

from config import TRAIN_DATA_PATH, TEST_DATA_PATH, BATCH_SIZE, CLASS_NAMES, NUM_CLASSES, INPUT_SHAPE


class DataGenerator(keras.utils.Sequence):
    def __init__(
            self, dataset_path: str, is_train: bool = True, batch_size: int = BATCH_SIZE,
            train_json_path: str = TRAIN_DATA_PATH, val_json_path: str = TEST_DATA_PATH,
            input_shape: Tuple[int, int, int] = INPUT_SHAPE, class_names: Tuple[str, str] = CLASS_NAMES,
            num_classes: int = NUM_CLASSES
    ) -> None:
        """
        Data generator for the task of hat/beard classifying.

        :param dataset_path: path to full dataset.
        :param is_train: if True, generating data from train_json_path and performing augmentation and every epoch
            shuffling. Else generating data from val_json_path without augmentation and every epoch shuffling.
        :param batch_size: batch size.
        :param train_json_path: local path to train json.
        :param val_json_path: local path to validation json.
        :param input_shape: input shape tuple (height, width, channels).
        :param class_names: tuple with class names.
        :param num_classes: number of classes.
        """
        self.dataset_path = dataset_path
        self.is_train = is_train
        self.batch_size = batch_size
        self.json_path = train_json_path if is_train else val_json_path
        self.input_shape = input_shape
        self.classes = class_names
        self.num_classes = num_classes

        with open(self.json_path) as f:
            self.data = json.load(f)
            np.random.shuffle(self.data)

        if is_train:
            augmentations = [
                aug.OneOf([aug.Crop(0.4, max_percent=0.1), aug.Crop(0.4, max_percent=0.075, two_side=True)], False),
                aug.Resize(self.input_shape),
                aug.ChangeContrast(0.35),
                aug.FlipLR(0.45),
                aug.Rotation(0.4, min_angle=-15, max_angle=15),
                aug.Scale(0.4, min_val=0.95),
                aug.OneOf([aug.GaussianBlur(0.4), aug.MedianBlur(0.4), aug.GaussianBlur(0.4)]),
                aug.OneOf([aug.AddUniformNoise(0.45), aug.MultUniformNoise(0.45), aug.Dropout(0.4)])
            ]
        else:
            augmentations = [aug.Resize(self.input_shape)]
        self.aug = aug.Sequential(augmentations)
        self.on_epoch_end()

    def on_epoch_end(self) -> None:
        """
        Random shuffling of training data at the end of each epoch.
        """
        if self.is_train:
            np.random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data) // self.batch_size + (len(self.data) % self.batch_size != 0)

    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Making batch.

        :param batch_idx: batch number.
        :return: image tensor and list with labels tensors for each output.
        """
        batch = self.data[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        images = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        labels_hat = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
        labels_beard = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
        for i, data_dict in enumerate(batch):
            image = cv2.cvtColor(cv2.imread(os.path.join(self.dataset_path, data_dict['path'])), cv2.COLOR_BGR2RGB)
            image = self.aug.augment(image)
            images[i, :, :, :] = image
            if data_dict['hat']:
                labels_hat[i, 0] = 1.0
            else:
                labels_hat[i, 1] = 1.0
            if data_dict['beard']:
                labels_beard[i, 0] = 1.0
            else:
                labels_beard[i, 1] = 1.0
        images = image_normalization(images)
        return np.float32(images), [np.float32(labels_hat), np.float32(labels_beard)]

    def show(self, batch_idx: int) -> None:
        """
        Method for showing original and augmented image with labels.

        :param batch_idx: batch number.
        """
        batch = self.data[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        for i, data_dict in enumerate(batch):
            if not data_dict['hat'] and not data_dict['beard']:
                class_name = self.classes[0]
            elif data_dict['hat'] and not data_dict['beard']:
                class_name = self.classes[1]
            elif not data_dict['hat'] and data_dict['beard']:
                class_name = self.classes[2]
            else:
                class_name = self.classes[3]

            image = cv2.cvtColor(cv2.imread(os.path.join(self.dataset_path, data_dict['path'])), cv2.COLOR_BGR2RGB)
            image_augmented = self.aug.augment(image.copy())
            image_augmented = image_normalization(image_augmented)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
            plt.subplot(axes[0])
            plt.imshow(image)
            plt.title('Original, class = "{}"'.format(class_name))
            plt.subplot(axes[1])
            plt.imshow(image_augmented)
            plt.title('Augmented, class = "{}"'.format(class_name))
            if plt.waitforbuttonpress(0):
                plt.close('all')
                raise SystemExit
            plt.close(fig)


def image_normalization(image: np.ndarray) -> np.ndarray:
    """
    Image normalization.

    :param image: image numpy array.
    :return: normalized image.
    """
    return image / 255.0


def test_data_generator(dataset_path: str, is_train: bool = True) -> None:
    """
    Function for testing data generator. Visualizing original and augmented images with labels.
    Mouse click to continue, press any button to exit.

    :param dataset_path: path to full dataset.
    :param is_train: if True, generating train data. Else generating test data.
    """
    data_gen = DataGenerator(dataset_path=dataset_path, is_train=is_train)
    for index, _ in enumerate(data_gen):
        data_gen.show(index)
