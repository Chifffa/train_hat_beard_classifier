import os
from cv_utils import get_logger

LOGGING_LEVEL = 'debug'
logger = get_logger('posters_cnn', LOGGING_LEVEL)

# Dataset parameters.
DATASET_PATH = 'hat_beard_dataset'
NO_HAT_BEARD_PATH = '00_no_hat_no_beard'
HAT_PATH = '01_hat'
BEARD_PATH = '02_beard'
HAT_BEARD_PATH = '03_hat_beard'
TRAIN_TEST_SPLIT = 0.1

# Paths to data, saved logs and weights.
DATA_PATH = 'data'
LOGS_DIR = 'logs'
WEIGHTS_PATH = 'weights'
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train_data.json')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test_data.json')

# Training parameters.
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 100

# Custom model parameters.
ALPHA = 1.0
REGULARIZATION = 0.0005
ACTIVATION_TYPE = 'leaky'

MODEL_TYPE = 'custom_resnet18'
CLASS_NAMES = ('No hat, no beard', 'Hat', 'Beard', 'Hat and beard')
# 2 classes per each output.
NUM_CLASSES = 2
INPUT_SHAPE = (128, 128, 3)
INPUT_NAME = 'input'
OUTPUT_NAMES = ['hat_output', 'beard_output']
