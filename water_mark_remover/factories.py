from keras.optimizers import Adam, SGD, RMSprop


def get_optimizer(name: str, **kwargs):
    if name == "adam":
        return Adam(**kwargs)
    elif name == "sgd":
        return SGD(**kwargs)
    elif name == "rmsprop":
        return RMSprop(**kwargs)
    else:
        raise ValueError(f"Optimizer {name} not supported.")
