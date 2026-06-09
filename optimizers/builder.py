import torch.optim as optim


def build_optimizer(cfg, model):
    optim_name = cfg["optimizer"]["name"]
    lr = cfg["optimizer"]["lr"]
    weight_decay = cfg["optimizer"].get("weight_decay", 0.0)

    if optim_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif optim_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif optim_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=cfg["optimizer"].get("momentum", 0.9),
            weight_decay=weight_decay
        )

    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")

    return optimizer