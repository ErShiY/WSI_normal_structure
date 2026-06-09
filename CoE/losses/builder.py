import torch.nn as nn


def build_loss(cfg):
    loss_name = cfg["loss"]["name"]

    if loss_name == "ce":
        criterion = nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    return criterion