import yaml
import numpy as np
import wandb

from water_mark_remover import load_and_prepare_data, build_discriminator, get_optimizer


CONFIG_PATH = "config/"
CONFIG_FILE_NAME = "pretrained_discriminator.yml"
SWEEP_CONFIG_FILE_NAME = "discriminator_sweep.yml"


def construct_dataset(wm_images: np.ndarray, nowm_images: np.ndarray):
    """Returns X, y (0=watermarked, 1=non-watermarked)"""
    x = np.concatenate((wm_images, nowm_images))
    y = np.concatenate((np.zeros(wm_images.shape[0]), np.ones(nowm_images.shape[0])))
    return x, y


def update_config(config: dict):
    """Update the configuration with wandb's hyperparameters"""
    for key in wandb.config.keys():
        if key in config:
            config[key] = wandb.config[key]
        elif any(key in config[subsection] for subsection in config):
            for subsection in config:
                if key in config[subsection]:
                    config[subsection][key] = wandb.config[key]


def train():
    with open(CONFIG_PATH + CONFIG_FILE_NAME, "r") as f:
        config = yaml.safe_load(f)

    # Update the configuration with wandb's hyperparameters
    update_config(config)

    # Initialize a new wandb run with the updated configuration
    wandb.init(project="watermark-remover", config=config)

    X_train, _, y_train, _, X_val, y_val = load_and_prepare_data(
        **config["load_and_prepare_data"]
    )
    X_train, y_train = construct_dataset(X_train, y_train)
    X_val, y_val = construct_dataset(X_val, y_val)

    discriminator = build_discriminator(**config["build_discriminator"])
    optimizer = get_optimizer(**config["optimizer"])
    discriminator.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )

    # Train the discriminator
    history = discriminator.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        **config["fit"],
        callbacks=[wandb.keras.WandbCallback()]
    )


def main():
    with open(CONFIG_PATH + SWEEP_CONFIG_FILE_NAME, "r") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_config, project="your_project_name")
    wandb.agent(sweep_id, train)


if __name__ == "__main__":
    main()
