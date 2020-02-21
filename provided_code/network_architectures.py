''' Neural net architectures '''

from tensorflow.python.keras.layers import Input, LeakyReLU, BatchNormalization, \
    Conv3D, concatenate, Activation, SpatialDropout3D, AveragePooling3D, Conv3DTranspose
from tensorflow.python.keras.models import Model


class DefineDoseFromCT:
    """This class defines the architecture for a U-NET and must be inherited by a child class that
    executes various functions like training or predicting"""

    def generator_convolution(self, x, number_of_filters, use_batch_norm=True):
        """Convolution block used for generator"""
        x = Conv3D(number_of_filters, self.filter_size, strides=self.stride_size, padding="same", use_bias=False)(x)
        if use_batch_norm:
            x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def generator_convolution_transpose(self, x, nodes, use_dropout=True, skip_x=None):
        """Convolution transpose block used for generator"""

        if skip_x is not None:
            x = concatenate([x, skip_x])
        x = Conv3DTranspose(nodes, self.filter_size, strides=self.stride_size, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        if use_dropout:
            x = SpatialDropout3D(0.2)(x)
        x = LeakyReLU(alpha=0)(x)  # Use LeakyReLU(alpha = 0) instead of ReLU because ReLU is buggy when saved

        return x

    def define_generator(self):
        """Makes a generator that takes a CT image as input to generate a dose distribution of the same dimensions"""

        # Define inputs
        ct_image = Input(self.ct_shape)
        roi_masks = Input(self.roi_masks_shape)

        # Build Model starting with Conv3D layers
        x = concatenate([ct_image, roi_masks])
        x1 = self.generator_convolution(x, self.initial_number_of_filters)
        x2 = self.generator_convolution(x1, 2 * self.initial_number_of_filters)
        x3 = self.generator_convolution(x2, 4 * self.initial_number_of_filters)
        x4 = self.generator_convolution(x3, 8 * self.initial_number_of_filters)
        x5 = self.generator_convolution(x4, 8 * self.initial_number_of_filters)
        x6 = self.generator_convolution(x5, 8 * self.initial_number_of_filters, use_batch_norm=False)

        # Build model back up from bottleneck
        x5b = self.generator_convolution_transpose(x6, 8 * self.initial_number_of_filters, use_dropout=False)
        x4b = self.generator_convolution_transpose(x5b, 8 * self.initial_number_of_filters, skip_x=x5)
        x3b = self.generator_convolution_transpose(x4b, 4 * self.initial_number_of_filters, use_dropout=False, skip_x=x4)
        x2b = self.generator_convolution_transpose(x3b, 2 * self.initial_number_of_filters, skip_x=x3)
        x1b = self.generator_convolution_transpose(x2b, self.initial_number_of_filters, use_dropout=False, skip_x=x2)

        # Final layer
        x0b = concatenate([x1b, x1])
        x0b = Conv3DTranspose(1, self.filter_size, strides=self.stride_size, padding="same")(x0b)
        x_final = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding="same")(x0b)
        final_dose = Activation("relu")(x_final)

        # Compile model for use
        self.generator = Model(inputs=[ct_image, roi_masks], outputs=final_dose, name="generator")
        self.generator.compile(loss="mean_absolute_error", optimizer=self.gen_optimizer)
        self.generator.summary()
