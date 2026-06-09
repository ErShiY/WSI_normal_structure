from dataset.builder import build_dataloader
from models.builder import build_model
from losses.builder import build_loss
from optimizers.builder import build_optimizer
from engine.trainer import Trainer


cfg = {
    "dataset": "tcga-brca",
    "task": "slide",
    "fold": 6,

    "path": {
        "csv": r"I:/MM_Six/data_splits_tcga_brca_folds",
        "pt": r"I:/TCGA-BRCA R50/pt_files",
        "h5": r""
    },

    "train": {
        "batch_size": 1,
        "num_workers": 0,
        "epochs": 5,
        "device": "cuda"
    },

    "models": {
        "name": "attention_mil",
        "input_dim": 1024,
        "hidden_dim": 512,
        "dropout": 0.25,
        "num_classes": 2
    },

    "loss": {
        "name": "ce"
    },

    "optimizer": {
        "name": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-5
    },

}

loaders = build_dataloader(cfg)

model = build_model(cfg)
criterion = build_loss(cfg)
optimizer = build_optimizer(cfg, model)

trainer = Trainer(
    cfg=cfg,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=loaders["train"],
    val_loader=loaders["val"]
)

trainer.fit()