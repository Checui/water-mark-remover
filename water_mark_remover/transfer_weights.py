from keras.layers import Conv2D
import numpy as np


def transfer_weights_to_unet(unet_model, autoencoder_model):
    """
    Transfer weights from an autoencoder model to a U-Net model.

    :param unet_model: The U-Net model to which weights are to be transferred.
    :param autoencoder_model: The pre-trained autoencoder model.
    """
    for unet_layer in unet_model.layers:
        if "concatenate" not in unet_layer.name and isinstance(unet_layer, Conv2D):
            # Find the corresponding layer in the autoencoder model
            autoencoder_layer = next(
                (layer for layer in autoencoder_model.layers if layer.name == unet_layer.name), None
            )

            if autoencoder_layer:
                # Extract weights from the autoencoder layer
                weights = autoencoder_layer.get_weights()

                # Calculate the number of filters in the autoencoder and U-Net layers
                filter_count = weights[0].shape[3]
                new_filter_count = unet_layer.get_weights()[0].shape[3]

                # Adjust the shape of the weights to accommodate additional feature maps from skip connections
                adjusted_weights = [
                    np.concatenate(
                        [
                            np.random.normal(
                                size=(
                                    weights[0].shape[0],
                                    weights[0].shape[1],
                                    weights[0].shape[2],
                                    new_filter_count - filter_count,
                                )
                            ),
                            weights[0],
                        ],
                        axis=3,
                    ),
                    np.concatenate(
                        [np.zeros(new_filter_count - filter_count), weights[1]], axis=0
                    ),
                ]

                # Set the adjusted weights to the U-Net layer
                unet_layer.set_weights(adjusted_weights)
