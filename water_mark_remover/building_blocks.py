
import keras
from keras.models import Model
from tensorflow import keras
from keras.models import Model
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Input,
    Conv2D,
    UpSampling2D,
    BatchNormalization,
)


def downsample_block(x, filters, kernel_size=(3, 3), padding="same", activation="relu"):
    # downsample
    conv1 = Conv2D(
        filters,
        kernel_size,
        activation="relu",
        padding=padding,
        kernel_initializer="he_normal",
    )(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 98

    return pool1


def generate_generator(width, height):
    x = Input(shape=(width, height, 3))  # 196

    # downsample
    conv1 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 98

    conv2 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # middle
    conv5 = Conv2D(
        1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(
        1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv5)
    conv5 = BatchNormalization()(conv5)

    # upsample
    up6 = Conv2D(
        512, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv5))
    up6 = BatchNormalization()(up6)
    merge6 = keras.layers.concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(
        256, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv6))
    up7 = BatchNormalization()(up7)
    merge7 = keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(
        128, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv7))
    up8 = BatchNormalization()(up8)
    merge8 = keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(
        64, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9)
    merge9 = keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(
        2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv9 = BatchNormalization()(conv9)

    # output
    output = Conv2D(3, 1, activation="sigmoid")(conv9)

    return Model(inputs=x, outputs=output)
