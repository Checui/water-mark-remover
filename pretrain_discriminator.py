import yaml
import numpy as np

from water_mark_remover import load_and_prepare_data, build_discriminator


CONFIG_PATH = "config/"
CONFIG_FILE_NAME = "example_config.yml"


def construct_dataset(wm_images: np.ndarray, nowm_images: np.ndarray):
    """Returns X, y (0=watermarked, 1=non-watermarked)"""
    x = np.concatenate((wm_images, nowm_images))
    y = np.concatenate((np.zeros(wm_images.shape[0]), np.ones(nowm_images.shape[0])))
    return x, y


def train(config: dict):
    X_train, X_test, y_train, y_test, X_val, y_val = load_and_prepare_data(
        **config["load_and_prepare_data"]
    )
    X_train, y_train = construct_dataset(X_train, y_train)
    X_test, y_test = construct_dataset(X_test, y_test)
    X_val, y_val = construct_dataset(X_val, y_val)

    discriminator = build_discriminator(**config["build_discriminator"])
    discriminator.compile(**config["compile"])

    # Train the discriminator
    discriminator.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        **config["fit"],
    )

    # Evaluate the discriminator
    discriminator.evaluate(X_test, y_test)


def main():
    # Read config file
    with open(CONFIG_PATH + CONFIG_FILE_NAME, "r") as f:
        config = yaml.safe_load(f)
    config


if __name__ == "__main__":
    main()
