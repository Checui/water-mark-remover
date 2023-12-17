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


def downsample_block(x, filters, kernel_size=(3, 3), padding="same", strides=(1, 1), use_batch_norm=True):
    """Create a downsample block with an optional Batch Normalization layer."""
    conv = Conv2D(
        filters,
        kernel_size,
        activation="relu",
        strides=strides,
        padding=padding,
        kernel_initializer="he_normal",
    )(x)

    if use_batch_norm:
        conv = BatchNormalization()(conv)

    conv = Conv2D(
        filters,
        kernel_size,
        activation="relu",
        padding=padding,
        kernel_initializer="he_normal",
    )(conv)

    if use_batch_norm:
        conv = BatchNormalization()(conv)

    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool



def upsample_block(x, skip_connection, filters, kernel_size=(3, 3), padding="same", use_batch_norm=True, use_concatenate=True):
    """Create an upsample block with optional Batch Normalization and Concatenate layers."""
    up = UpSampling2D(size=(2, 2))(x)

    if use_batch_norm:
        up = BatchNormalization()(up)

    if use_concatenate and skip_connection is not None:
        up = concatenate([skip_connection, up], axis=3)

    conv = Conv2D(
        filters,
        kernel_size,
        activation="relu",
        padding=padding,
        kernel_initializer="he_normal",
    )(up)

    if use_batch_norm:
        conv = BatchNormalization()(conv)

    conv = Conv2D(
        filters,
        kernel_size,
        activation="relu",
        padding=padding,
        kernel_initializer="he_normal",
    )(conv)

    if use_batch_norm:
        conv = BatchNormalization()(conv)

    return conv




def middle_block(x, filters, kernel_size=(3, 3), padding="same", use_batch_norm=True):
    """Create the middle block of the network with optional Batch Normalization."""
    conv = Conv2D(
        filters,
        kernel_size,
        activation="relu",
        padding=padding,
        kernel_initializer="he_normal",
    )(x)

    if use_batch_norm:
        conv = BatchNormalization()(conv)

    conv = Conv2D(
        filters,
        kernel_size,
        activation="relu",
        padding=padding,
        kernel_initializer="he_normal",
    )(conv)

    if use_batch_norm:
        conv = BatchNormalization()(conv)

    return conv



def build_generator(width, height, filter_sizes, use_batch_norm=True, use_skip_connections=True):
    """Build the generator model with dynamic number of blocks and filter sizes."""
    if len(filter_sizes) < 2:
        raise ValueError("filter_sizes must have at least two elements for downsampling and upsampling stages.")

    inputs = Input(shape=(width, height, 3))

    # Dynamic Downsampling
    skip_connections = []
    x = inputs
    for filters in filter_sizes[:-1]:
        d, p = downsample_block(x, filters, use_batch_norm=use_batch_norm)
        skip_connections.append(d)
        x = p

    # Middle block
    middle_filters = filter_sizes[-1]
    middle = middle_block(x, middle_filters, use_batch_norm=use_batch_norm)

    # Dynamic Upsampling
    x = middle
    for filters, skip_connection in zip(reversed(filter_sizes[:-1]), reversed(skip_connections)):
        if use_skip_connections:
            x = upsample_block(x, skip_connection, filters, use_batch_norm=use_batch_norm)
        else:
            x = upsample_block(x, None, filters, use_batch_norm=use_batch_norm)

    # Output
    output = Conv2D(3, (1, 1), activation="sigmoid")(x)

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

    return Model(inputs=inputs, outputs=[output, inputs])
