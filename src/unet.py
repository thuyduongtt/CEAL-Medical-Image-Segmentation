from __future__ import print_function

import cv2
import numpy as np
from keras import backend as K
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, SpatialDropout2D, Layer, \
    Conv2DTranspose, BatchNormalization, Cropping2D
from keras.models import Model
from keras.optimizers import Adam

from constants import img_rows, img_cols

# K.set_image_dim_ordering('th') # Theano dimension ordering in this code
K.set_image_data_format('channels_first')

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# Override Dropout. Make it able at test time.
def call(self, inputs, training=None):
    if 0. < self.rate < 1.:
        noise_shape = self._get_noise_shape(inputs)

        def dropped_inputs():
            return K.dropout(inputs, self.rate, noise_shape,
                             seed=self.seed)

        if training:
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        else:
            return K.in_test_phase(dropped_inputs, inputs, training=None)
    return inputs


Dropout.call = call


def get_unet(dropout):
    return vanilla_unet((1, img_rows, img_cols), dropout=dropout)

    inputs = Input((1, img_rows, img_cols))
    inputs = PrintLayer()(inputs, 'inputs')  # (None, 1, 192, 240)

    conv1 = Conv2D(32, 3, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv1 = PrintLayer()(conv1, 'conv1')  # (None, 32, 22, 27)

    conv2 = Conv2D(64, 3, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv2 = PrintLayer()(conv2, 'conv2')  # (None, 64, 2, 2)

    conv3 = Conv2D(128, 3, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)

    conv3 = PrintLayer()(conv3, 'conv3')  # (None, 128, 1, 1)

    conv4 = Conv2D(256, 3, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)

    conv4 = PrintLayer()(conv4, 'conv4')  # (None, 256, 1, 1)

    conv5 = Conv2D(512, 3, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, 3, 3, activation='relu', padding='same')(conv5)

    if dropout:
        conv5 = Dropout(0.5)(conv5)

    conv5 = PrintLayer()(conv5, 'conv5')  # (None, 512, 1, 1)

    up6 = Conv2DTranspose(256, 2, 2, padding='valid')(conv5)
    merge6 = concatenate([up6, conv4], axis=1)

    conv6 = Conv2D(256, 3, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(256, 3, 3, activation='relu', padding='same')(conv6)

    conv6 = PrintLayer()(conv6, 'conv6')

    up7 = Conv2DTranspose(128, 2, 2, padding='valid')(conv6)
    merge7 = concatenate([up7, conv3], axis=1)

    conv7 = Conv2D(128, 3, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(128, 3, 3, activation='relu', padding='same')(conv7)

    conv7 = PrintLayer()(conv7, 'conv7')

    up8 = Conv2DTranspose(64, 2, 2, padding='valid')(conv7)
    merge8 = concatenate([up8, conv2], axis=1)

    conv8 = Conv2D(64, 3, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(64, 3, 3, activation='relu', padding='same')(conv8)

    conv8 = PrintLayer()(conv8, 'conv8')

    up9 = Conv2DTranspose(32, 2, 2, padding='valid')(conv8)
    merge9 = concatenate([up9, conv1], axis=1)

    conv9 = Conv2D(32, 3, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(32, 3, 3, activation='relu', padding='same')(conv9)
    # conv9 = Conv2D(2, 3, 3, activation='relu', padding='same')(conv9)

    conv9 = PrintLayer()(conv9, 'conv9')

    conv10 = Conv2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def vanilla_unet(
        input_shape,
        num_classes=1,
        dropout=0.5,
        filters=32,
        num_layers=4,
        output_activation='sigmoid'):  # 'sigmoid' or 'softmax'

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='valid')
        x = PrintLayer()(x, f'down {filters}')
        down_layers.append(x)
        x = MaxPooling2D((2, 2), strides=2)(x)
        filters = filters * 2  # double the number of filters with each layer

    x = Dropout(dropout)(x)
    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='valid')

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='valid')(x)

        ch, cw = get_crop_shape(K.int_shape(conv), K.int_shape(x))
        conv = Cropping2D(cropping=(ch, cw))(conv)

        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='valid')
        x = PrintLayer()(x, f'up {filters}')

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def conv2d_block(
        inputs,
        use_batch_norm=True,
        dropout=0.3,
        dropout_type="spatial",
        filters=16,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
):
    if dropout_type == "spatial":
        DO = SpatialDropout2D
    elif dropout_type == "standard":
        DO = Dropout
    else:
        raise ValueError(
            f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
        )

    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = DO(dropout)(c)
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = target[2] - refer[2]
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = target[1] - refer[1]
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)


class PrintLayer(Layer):
    def __call__(self, x, title=None):
        if title is not None:
            print(title)
        print(x.shape)
        return x
