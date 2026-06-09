import argparse

from utils.config import load_config
from dataset.builder import build_dataloader
from models.builder import build_model
from losses.builder import build_loss
from optimizers.builder import build_optimizer
from engine.trainer import Trainer
from schedulers.builder import build_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="WSI Training Pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/tcga_brca.yaml",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. load config
    cfg = load_config(args.config)

    print(f"Using config: {args.config}")
    print(f"Dataset: {cfg['dataset']}")
    print(f"Model: {cfg['model']['name']}")

    # 2. build dataloader
    loaders = build_dataloader(cfg)

    train_loader = loaders["train"]
    val_loader = loaders.get("val", None)
    test_loader = loaders.get("test", None)

    # 3. build model
    model = build_model(cfg)

    # 4. build loss
    criterion = build_loss(cfg)

    # 5. build optimizer
    optimizer = build_optimizer(cfg, model)

    scheduler = build_scheduler(cfg, optimizer)

    # 6. build trainer
    trainer = Trainer(
        cfg=cfg,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader = test_loader
    )

    # 7. start training
    trainer.fit()


if __name__ == "__main__":
    main()