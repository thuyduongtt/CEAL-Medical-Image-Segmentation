from __future__ import print_function

import cv2
import numpy as np
from keras import backend as K
from keras.layers import Input, concatenate, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, Layer
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
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', padding='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv1 = PrintLayer()(conv1, 'conv1')

    conv2 = Convolution2D(64, 3, 3, activation='relu', padding='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv2 = PrintLayer()(conv2, 'conv2')

    conv3 = Convolution2D(128, 3, 3, activation='relu', padding='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)

    conv3 = PrintLayer()(conv3, 'conv3')

    conv4 = Convolution2D(256, 3, 3, activation='relu', padding='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)

    conv4 = PrintLayer()(conv4, 'conv4')

    conv5 = Convolution2D(512, 3, 3, activation='relu', padding='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', padding='same')(conv5)

    if dropout:
        conv5 = Dropout(0.5)(conv5)

    conv5 = PrintLayer()(conv5, 'conv5')

    up6 = Convolution2D(256, 2, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([up6, conv4], axis=1)

    conv6 = Convolution2D(256, 3, 3, activation='relu', padding='same')(merge6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', padding='same')(conv6)

    conv6 = PrintLayer()(conv6, 'conv6')

    up7 = Convolution2D(128, 2, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([up7, conv3], axis=1)

    conv7 = Convolution2D(128, 3, 3, activation='relu', padding='same')(merge7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', padding='same')(conv7)

    conv7 = PrintLayer()(conv7, 'conv7')

    up8 = Convolution2D(64, 2, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([up8, conv2], axis=1)

    conv8 = Convolution2D(64, 3, 3, activation='relu', padding='same')(merge8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', padding='same')(conv8)

    conv8 = PrintLayer()(conv8, 'conv8')

    up9 = Convolution2D(32, 2, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([up9, conv1], axis=1)

    conv9 = Convolution2D(32, 3, 3, activation='relu', padding='same')(merge9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', padding='same')(conv9)

    conv9 = PrintLayer()(conv9, 'conv9')

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


class PrintLayer(Layer):
    def __call__(self, x, title=None):
        if title is not None:
            print(title)
        print(x.shape)
        return x
