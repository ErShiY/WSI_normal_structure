import torch.optim.lr_scheduler as lr_scheduler


def build_scheduler(cfg, optimizer):
    scheduler_cfg = cfg.get("scheduler", None)

    if scheduler_cfg is None:
        return None

    scheduler_name = scheduler_cfg["name"]

    if scheduler_name == "steplr":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 50),
            gamma=scheduler_cfg.get("gamma", 0.1)
        )

    elif scheduler_name == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("T_max", cfg["train"]["epochs"]),
            eta_min=scheduler_cfg.get("eta_min", 0.0)
        )

    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return scheduler