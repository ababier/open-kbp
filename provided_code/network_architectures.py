""" Neural net architectures """
from typing import Optional

from keras.engine.keras_tensor import KerasTensor
from keras.layers import Activation, AveragePooling3D, Conv3D, Conv3DTranspose, Input, LeakyReLU, SpatialDropout3D, concatenate
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.models import Model
from keras.optimizers.optimizer_v2.optimizer_v2 import OptimizerV2

from provided_code.data_shapes import DataShapes


class DefineDoseFromCT:
    """This class defines the architecture for a U-NET and must be inherited by a child class that
    executes various functions like training or predicting"""

    def __init__(
        self,
        data_shapes: DataShapes,
        initial_number_of_filters: int,
        filter_size: tuple[int, int, int],
        stride_size: tuple[int, int, int],
        gen_optimizer: OptimizerV2,
    ):
        self.data_shapes = data_shapes
        self.initial_number_of_filters = initial_number_of_filters
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.gen_optimizer = gen_optimizer

    def make_convolution_block(self, x: KerasTensor, num_filters: int, use_batch_norm: bool = True) -> KerasTensor:
        x = Conv3D(num_filters, self.filter_size, strides=self.stride_size, padding="same", use_bias=False)(x)
        if use_batch_norm:
            x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def make_convolution_transpose_block(
        self, x: KerasTensor, num_filters: int, use_dropout: bool = True, skip_x: Optional[KerasTensor] = None
    ) -> KerasTensor:
        if skip_x is not None:
            x = concatenate([x, skip_x])
        x = Conv3DTranspose(num_filters, self.filter_size, strides=self.stride_size, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        if use_dropout:
            x = SpatialDropout3D(0.2)(x)
        x = LeakyReLU(alpha=0)(x)  # Use LeakyReLU(alpha = 0) instead of ReLU because ReLU is buggy when saved
        return x

    def define_generator(self) -> Model:
        """Makes a generator that takes a CT image as input to generate a dose distribution of the same dimensions"""

        # Define inputs
        ct_image = Input(self.data_shapes.ct)
        roi_masks = Input(self.data_shapes.structure_masks)

        # Build Model starting with Conv3D layers
        x = concatenate([ct_image, roi_masks])
        x1 = self.make_convolution_block(x, self.initial_number_of_filters)
        x2 = self.make_convolution_block(x1, 2 * self.initial_number_of_filters)
        x3 = self.make_convolution_block(x2, 4 * self.initial_number_of_filters)
        x4 = self.make_convolution_block(x3, 8 * self.initial_number_of_filters)
        x5 = self.make_convolution_block(x4, 8 * self.initial_number_of_filters)
        x6 = self.make_convolution_block(x5, 8 * self.initial_number_of_filters, use_batch_norm=False)

        # Build model back up from bottleneck
        x5b = self.make_convolution_transpose_block(x6, 8 * self.initial_number_of_filters, use_dropout=False)
        x4b = self.make_convolution_transpose_block(x5b, 8 * self.initial_number_of_filters, skip_x=x5)
        x3b = self.make_convolution_transpose_block(x4b, 4 * self.initial_number_of_filters, use_dropout=False, skip_x=x4)
        x2b = self.make_convolution_transpose_block(x3b, 2 * self.initial_number_of_filters, skip_x=x3)
        x1b = self.make_convolution_transpose_block(x2b, self.initial_number_of_filters, use_dropout=False, skip_x=x2)

        # Final layer
        x0b = concatenate([x1b, x1])
        x0b = Conv3DTranspose(1, self.filter_size, strides=self.stride_size, padding="same")(x0b)
        x_final = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding="same")(x0b)
        final_dose = Activation("relu")(x_final)

        # Compile model for use
        generator = Model(inputs=[ct_image, roi_masks], outputs=final_dose, name="generator")
        generator.compile(loss="mean_absolute_error", optimizer=self.gen_optimizer)
        generator.summary()
        return generator
