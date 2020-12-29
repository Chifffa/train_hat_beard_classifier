from typing import Tuple, Optional

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D, Input, Add
from tensorflow.python.framework.ops import Tensor
from cv_utils import Activation


class CustomResNet18:
    def __init__(
            self, input_shape: Tuple[int, int, int], num_classes: int, alpha: float, regularization: Optional[float],
            activation_type: str, input_name: str
    ) -> None:
        """
        Custom implementation of the stripped-down ResNet18 for hat/beard classification task.
        This implementation has 2 output layers.

        :param input_shape: input shape (height, width, channels).
        :param num_classes: number of classes.
        :param alpha: network expansion factor, determines the number of filters in each layer.
        :param regularization: pass float < 1 (good is 0.0005) to make regularization or None to ignore it.
        :param activation_type: type of activation function. See cv_utils.Activation.
        :param input_name: name of the input tensor.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation_type = activation_type
        self.input_name = input_name

        self.init_filters = int(16 * alpha)
        self.ker_reg = None if regularization is None else keras.regularizers.l2(regularization)

    def build(self) -> keras.models.Model:
        """
        Building CNN model for classification task.

        :return: keras.model.Model() object.
        """
        inputs = Input(shape=self.input_shape, name=self.input_name)
        x = BatchNormalization()(inputs)
        x = Conv2D(self.init_filters, (3, 3), strides=2, use_bias=False, kernel_regularizer=self.ker_reg)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation_type)(x)

        x = self.res_block(x, self.init_filters * 2, 2)
        for _ in range(2):
            x = self.res_block(x, self.init_filters * 2)

        x = self.res_block(x, self.init_filters * 2 ** 2, 2)
        for _ in range(2):
            x = self.res_block(x, self.init_filters * 2 ** 2)

        x = self.res_block(x, self.init_filters * 2 ** 3, 2)
        for _ in range(2):
            x = self.res_block(x, self.init_filters * 2 ** 3)

        x1 = self.bottleneck_output(x, 'hat')
        x2 = self.bottleneck_output(x, 'beard')
        return keras.models.Model(inputs=inputs, outputs=[x1, x2])

    def res_block(self, x: Tensor, filters: int, stride: int = 1) -> Tensor:
        """
        Residual block. If stride == 1, then there are no any transformations in one of the branches.
        If stride > 1, then there are convolution with 1x1 filters in one of the branches.

        :param x: input tensor.
        :param filters: number of filters in output tensor.
        :param stride: convolution stride.
        :return: output tensor.
        """
        conv_kwargs = {'use_bias': False, 'padding': 'same', 'kernel_regularizer': self.ker_reg}
        x1 = Conv2D(filters, (3, 3), strides=stride, **conv_kwargs)(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation(self.activation_type)(x1)
        x1 = Conv2D(filters, (3, 3), **conv_kwargs)(x1)
        if stride == 1:
            x2 = x
        else:
            x2 = Conv2D(filters, (1, 1), strides=stride, **conv_kwargs)(x)
        x_out = Add()([x1, x2])
        x_out = BatchNormalization()(x_out)
        x_out = Activation(self.activation_type)(x_out)
        return x_out

    def bottleneck_output(self, x: Tensor, name: str) -> Tensor:
        """
        Creating output as bottleneck with 2 convolutional layers, global average pooling and dense.

        :param x: input tensor.
        :param name: name of the current output.
        :return: output tensor.
        """
        conv_kwargs = {'use_bias': False, 'padding': 'same', 'kernel_regularizer': self.ker_reg}
        x = Conv2D(self.init_filters * 2 ** 2, (3, 3), **conv_kwargs)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation_type)(x)
        x = Conv2D(self.init_filters * 2 ** 3, (3, 3), **conv_kwargs)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation_type)(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(self.num_classes, kernel_regularizer=self.ker_reg, activation='softmax', name=name)(x)
        return x
