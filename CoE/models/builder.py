from .mil_model import AttentionMIL


def get_num_classes(cfg):
    dataset = cfg["dataset"]
    task = cfg.get("task", "slide")

    if dataset == "camelyon17":
        if task == "slide":
            return 4
        elif task == "patient":
            return 5
        else:
            raise ValueError(f"Unsupported task for camelyon17: {task}")

    elif dataset in ["tcga-brca", "tcga-nsclc"]:
        return 2

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def build_model(cfg):
    model_name = cfg["model"]["name"]
    num_classes = get_num_classes(cfg)

    if model_name == "attention_mil":
        model = AttentionMIL(
            input_dim=cfg["model"]["input_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            num_classes=num_classes,
            dropout=cfg["model"].get("dropout", 0.25)
        )
    # elif model_name == "coe_mil":
    #     models =

    else:
        raise ValueError(f"Unsupported models: {model_name}")

    return model
