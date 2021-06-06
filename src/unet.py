from __future__ import print_function

from keras import backend as K
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, MaxPool2D, UpSampling2D, Dropout, Layer, \
    Concatenate
from keras.models import Model
from keras.optimizers import Adam

from constants import img_rows, img_cols, n_channel

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


def get_unet_old(dropout):
    inputs = Input((1, img_rows, img_cols))
    inputs = PrintLayer()(inputs, 'inputs')  # (None, 1, 192, 240)

    conv1 = Conv2D(32, 3, 3, activation='relu', padding='same')(inputs)
    conv1 = PrintLayer()(conv1, 'conv1')
    conv1 = Conv2D(32, 3, 3, activation='relu', padding='same')(conv1)
    conv1 = PrintLayer()(conv1, 'conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv1 = PrintLayer()(conv1, 'conv1')  # (None, 32, 22, 27)

    conv2 = Conv2D(64, 3, 3, activation='relu', padding='same')(pool1)
    conv2 = PrintLayer()(conv2, 'conv2')
    conv2 = Conv2D(64, 3, 3, activation='relu', padding='same')(conv2)
    conv2 = PrintLayer()(conv2, 'conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv2 = PrintLayer()(conv2, 'conv2')  # (None, 64, 2, 2)

    conv3 = Conv2D(128, 3, 3, activation='relu', padding='same')(pool2)
    conv3 = PrintLayer()(conv3, 'conv3')
    conv3 = Conv2D(128, 3, 3, activation='relu', padding='same')(conv3)
    conv3 = PrintLayer()(conv3, 'conv3')
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)

    conv3 = PrintLayer()(conv3, 'conv3')  # (None, 128, 1, 1)

    conv4 = Conv2D(256, 3, 3, activation='relu', padding='same')(pool3)
    conv4 = PrintLayer()(conv4, 'conv4')
    conv4 = Conv2D(256, 3, 3, activation='relu', padding='same')(conv4)
    conv4 = PrintLayer()(conv4, 'conv4')
    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)

    conv4 = PrintLayer()(conv4, 'conv4')  # (None, 256, 1, 1)

    conv5 = Conv2D(512, 3, 3, activation='relu', padding='same')(pool4)
    conv5 = PrintLayer()(conv5, 'conv5')
    conv5 = Conv2D(512, 3, 3, activation='relu', padding='same')(conv5)
    conv5 = PrintLayer()(conv5, 'conv5')

    if dropout:
        conv5 = Dropout(0.5)(conv5)

    conv5 = PrintLayer()(conv5, 'conv5')  # (None, 512, 1, 1)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = PrintLayer()(up6, 'up6')
    merge6 = concatenate([up6, conv4], axis=3)
    merge6 = PrintLayer()(merge6, 'merge6')

    conv6 = Conv2D(256, 3, 3, activation='relu', padding='same')(merge6)
    conv6 = PrintLayer()(conv6, 'conv6')
    conv6 = Conv2D(256, 3, 3, activation='relu', padding='same')(conv6)
    conv6 = PrintLayer()(conv6, 'conv6')

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = PrintLayer()(up7, 'up7')
    merge7 = concatenate([up7, conv3], axis=3)
    merge7 = PrintLayer()(merge7, 'merge7')

    conv7 = Conv2D(128, 3, 3, activation='relu', padding='same')(merge7)
    conv7 = PrintLayer()(conv7, 'conv7')
    conv7 = Conv2D(128, 3, 3, activation='relu', padding='same')(conv7)
    conv7 = PrintLayer()(conv7, 'conv7')

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = PrintLayer()(up8, 'up8')
    merge8 = concatenate([up8, conv2], axis=3)
    merge8 = PrintLayer()(merge8, 'merge8')

    conv8 = Conv2D(64, 3, 3, activation='relu', padding='same')(merge8)
    conv8 = PrintLayer()(conv8, 'conv8')
    conv8 = Conv2D(64, 3, 3, activation='relu', padding='same')(conv8)
    conv8 = PrintLayer()(conv8, 'conv8')

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = PrintLayer()(up9, 'up9')
    merge9 = concatenate([up9, conv1], axis=3)
    merge9 = PrintLayer()(merge9, 'merge9')

    conv9 = Conv2D(32, 3, 3, activation='relu', padding='same')(merge9)
    conv9 = PrintLayer()(conv9, 'conv9')
    conv9 = Conv2D(32, 3, 3, activation='relu', padding='same')(conv9)
    conv9 = PrintLayer()(conv9, 'conv9')
    # conv9 = Conv2D(2, 3, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def get_unet(dropout):
    filters = 32
    input_layer = Input(shape=[n_channel, img_rows, img_cols])
    layers = [input_layer]
    residuals = []

    # Down 1, 128
    d1, res1 = down(input_layer, filters)
    residuals.append(res1)

    filters *= 2

    # Down 2, 64
    d2, res2 = down(d1, filters)
    residuals.append(res2)

    filters *= 2

    # Down 3, 32
    d3, res3 = down(d2, filters)
    residuals.append(res3)

    filters *= 2

    # Down 4, 16
    d4, res4 = down(d3, filters)
    residuals.append(res4)

    filters *= 2

    # Down 5, 8
    d5 = down(d4, filters, pool=False)

    if dropout:
        d5 = Dropout(0.5)(d5)

    # Up 1, 16
    up1 = up(d5, residual=residuals[-1], filters=filters / 2)

    filters /= 2

    # Up 2,  32
    up2 = up(up1, residual=residuals[-2], filters=filters / 2)

    filters /= 2

    # Up 3, 64
    up3 = up(up2, residual=residuals[-3], filters=filters / 2)

    filters /= 2

    # Up 4, 128
    up4 = up(up3, residual=residuals[-4], filters=filters / 2)

    out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)

    model = Model(input_layer, out)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    # model.summary()

    return model


def down(input_layer, filters, pool=True):
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
    residual = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    if pool:
        max_pool = MaxPool2D()(residual)
        return max_pool, residual
    else:
        return residual


def up(input_layer, residual, filters):
    filters = int(filters)
    upsample = UpSampling2D()(input_layer)
    upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
    concat = Concatenate(axis=1)([residual, upconv])
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat)
    conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    return conv2


class PrintLayer(Layer):
    def __call__(self, x, title=None):
        if title is not None:
            print(title)
        print(x.shape)
        return x
