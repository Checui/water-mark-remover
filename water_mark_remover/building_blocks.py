from keras.models import Model
from keras.models import Model
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Input,
    Conv2D,
    UpSampling2D,
    BatchNormalization,
    concatenate,
    Flatten,
    Dense,
    LeakyReLU,
    Dropout,
    Activation,
)


def downsample_block(x, filters, kernel_size=(3, 3), padding="same", strides=(1, 1)):
    """Create a downsample block."""
    conv = Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer="he_normal",
    )(x)
    conv = BatchNormalization()(conv)
    conv = Conv2D(
        filters,
        kernel_size,
        activation="relu",
        padding=padding,
        kernel_initializer="he_normal",
    )(conv)
    conv = BatchNormalization()(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool


def upsample_block(x, skip_connection, filters, kernel_size=(3, 3), padding="same"):
    """Create an upsample block."""
    up = Conv2D(
        filters,
        kernel_size=(2, 2),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(UpSampling2D(size=(2, 2))(x))
    up = BatchNormalization()(up)
    merge = concatenate([skip_connection, up], axis=3)
    conv = Conv2D(
        filters,
        kernel_size,
        activation="relu",
        padding=padding,
        kernel_initializer="he_normal",
    )(merge)
    conv = BatchNormalization()(conv)
    conv = Conv2D(
        filters,
        kernel_size,
        activation="relu",
        padding=padding,
        kernel_initializer="he_normal",
    )(conv)
    conv = BatchNormalization()(conv)
    return conv


def middle_block(x, filters, kernel_size=(3, 3), padding="same"):
    """Create the middle block of the network."""
    conv = Conv2D(
        filters,
        kernel_size,
        activation="relu",
        padding=padding,
        kernel_initializer="he_normal",
    )(x)
    conv = BatchNormalization()(conv)
    conv = Conv2D(
        filters,
        kernel_size,
        activation="relu",
        padding=padding,
        kernel_initializer="he_normal",
    )(conv)
    conv = BatchNormalization()(conv)
    return conv


def build_generator(width, height):
    """Build the generator model."""
    inputs = Input(shape=(width, height, 3))

    # Downsample
    d1, p1 = downsample_block(inputs, 64)
    d2, p2 = downsample_block(p1, 128)
    d3, p3 = downsample_block(p2, 256)
    d4, p4 = downsample_block(p3, 512)

    # Middle
    middle = middle_block(p4, 1024)

    # Upsample
    u1 = upsample_block(middle, d4, 512)
    u2 = upsample_block(u1, d3, 256)
    u3 = upsample_block(u2, d2, 128)
    u4 = upsample_block(u3, d1, 64)

    # Output
    output = Conv2D(3, (1, 1), activation="sigmoid")(u4)

    return Model(inputs, output)


def build_discriminator(
    input_shape,
    filters,
    kernel_size,
    strides,
    activation="leaky_relu",
    alpha=0.2,
    dropout_rate=0.3,
    use_batch_norm=True,
    dense_units=512,
    final_activation="sigmoid",
):
    """Builds the discriminator model."""

    if (
        len(filters) != len(strides)
    ):
        raise ValueError(
            f"The length of filters ({len(filters)})and strides ({len(strides)}) must be equal."
        )

    # Input Layer
    inputs = Input(shape=input_shape)

    x = inputs

    # Convolutional layers
    for n_filters, stride in zip(filters, strides):
        x = Conv2D(
            n_filters,
            kernel_size,
            strides=stride,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        if activation == "leaky_relu":
            x = LeakyReLU(alpha=alpha)(x)
        else:
            x = Activation(activation)(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(dense_units, activation="relu")(x)

    # Output layer
    output = Dense(1, activation=final_activation)(x)

    return Model(inputs, output)
