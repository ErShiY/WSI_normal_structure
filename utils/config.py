import os
import yaml


def load_config(config_path):
    """
    Load yaml config file.

    Args:
        config_path: path to yaml config file

    Returns:
        cfg: dict
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config file is empty: {config_path}")

    check_config(cfg)

    return cfg


def check_config(cfg):
    """
    Basic config validation.
    Only check necessary fields for current training pipeline.
    """

    required_keys = [
        "dataset",
        "task",
        "fold",
        "path",
        "train",
        "model",
        "loss",
        "optimizer"
    ]

    for key in required_keys:
        if key not in cfg:
            raise KeyError(f"Missing required config key: {key}")

    path_keys = ["csv"]

    if cfg["dataset"] == "camelyon17":
        path_keys.append("h5")
    elif cfg["dataset"] in ["tcga-brca", "tcga-nsclc"]:
        path_keys.append("pt")
    else:
        raise ValueError(f"Unsupported dataset: {cfg['dataset']}")

    for key in path_keys:
        if key not in cfg["path"]:
            raise KeyError(f"Missing path config: path.{key}")

    if "name" not in cfg["model"]:
        raise KeyError("Missing model config: model.name")

    if "name" not in cfg["loss"]:
        raise KeyError("Missing loss config: loss.name")

    if "name" not in cfg["optimizer"]:
        raise KeyError("Missing optimizer config: optimizer.name")

    if "lr" not in cfg["optimizer"]:
        raise KeyError("Missing optimizer config: optimizer.lr")