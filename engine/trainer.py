import csv
import os
import torch
from utils.metrics import compute_classification_metrics
from utils.logger import MetricLogger
from utils.early_stopping import EarlyStopping


class Trainer:
    def __init__(self, cfg, model, criterion, optimizer, scheduler, train_loader, val_loader=None, test_loader=None):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.checkpoint_cfg = cfg.get("checkpoint", {})
        self.save_dir = self.checkpoint_cfg.get("save_dir", "outputs/checkpoints")
        self.monitor = self.checkpoint_cfg.get("monitor", "macro_f1")
        self.mode = self.checkpoint_cfg.get("mode", "max")
        self.save_best = self.checkpoint_cfg.get("save_best", True)

        os.makedirs(self.save_dir, exist_ok=True)

        if self.mode == "max":
            self.best_metric = -float("inf")
        elif self.mode == "min":
            self.best_metric = float("inf")
        else:
            raise ValueError(f"Unsupported checkpoint mode: {self.mode}")

        log_cfg = cfg.get("log", {})
        log_save_dir = log_cfg.get("save_dir", "outputs/logs")
        log_filename = log_cfg.get("filename", "train_log.csv")

        self.logger = MetricLogger(
            save_dir=log_save_dir,
            filename=log_filename
        )

        early_cfg = cfg.get("early_stopping", {})
        self.use_early_stopping = early_cfg.get("enable", False)

        if self.use_early_stopping:
            self.early_stopping = EarlyStopping(
                monitor=early_cfg.get("monitor", "macro_f1"),
                mode=early_cfg.get("mode", "max"),
                patience=early_cfg.get("patience", 8),
                min_delta=early_cfg.get("min_delta", 0.0)
            )
        else:
            self.early_stopping = None

        result_cfg = cfg.get("results", {})
        self.result_save_dir = result_cfg.get("save_dir", "outputs/results")
        self.result_filename = result_cfg.get("filename", "test_results.csv")

        os.makedirs(self.result_save_dir, exist_ok=True)

        self.device = torch.device(
            cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model.to(self.device)

    def move_batch_to_device(self, batch):
        new_batch = {}

        for key, value in batch.items():
            if torch.is_tensor(value):
                new_batch[key] = value.to(self.device)
            else:
                new_batch[key] = value

        return new_batch

    def train_one_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for step, batch in enumerate(self.train_loader):
            batch = self.move_batch_to_device(batch)

            features = batch["features"]
            labels = batch["slide_label"].long()

            outputs = self.model(features)
            logits = outputs["logits"]

            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples

        print(
            f"Epoch [{epoch}] "
            f"Train Loss: {avg_loss:.4f} "
            f"Train Acc: {acc:.4f}"
        )

        return {
            "loss": avg_loss,
            "acc": acc
        }

    @torch.no_grad()
    def validate(self, epoch):
        if self.val_loader is None:
            return None

        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_labels = []
        all_preds = []

        for batch in self.val_loader:
            batch = self.move_batch_to_device(batch)

            features = batch["features"]
            labels = batch["slide_label"].long()

            outputs = self.model(features)
            logits = outputs["logits"]

            loss = self.criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

        avg_loss = total_loss / total_samples
        metrics = compute_classification_metrics(all_labels, all_preds)

        print(
            f"Epoch [{epoch}] "
            f"Val Loss: {avg_loss:.4f} "
            f"Val Acc: {metrics['acc']:.4f} "
            f"Val Precision: {metrics['precision']:.4f} "
            f"Val Recall: {metrics['recall']:.4f} "
            f"Val Macro-F1: {metrics['macro_f1']:.4f}"
        )

        metrics["loss"] = avg_loss
        return metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "cfg": self.cfg
        }

        if is_best:
            save_path = os.path.join(self.save_dir, "best_model.pth")
        else:
            save_path = os.path.join(self.save_dir, f"epoch_{epoch}.pth")

        torch.save(checkpoint, save_path)

        print(f"Checkpoint saved to: {save_path}")

    def is_best_metric(self, current_metric):
        if self.mode == "max":
            return current_metric > self.best_metric
        elif self.mode == "min":
            return current_metric < self.best_metric
        else:
            raise ValueError(f"Unsupported checkpoint mode: {self.mode}")

    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def fit(self):
        num_epochs = self.cfg["train"]["epochs"]

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_one_epoch(epoch)

            if self.val_loader is not None:
                val_metrics = self.validate(epoch)

                current_metric = val_metrics[self.monitor]

                log_row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["acc"]
                }

                if self.val_loader is not None:
                    log_row.update({
                        "val_loss": val_metrics["loss"],
                        "val_acc": val_metrics["acc"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                        "val_macro_f1": val_metrics["macro_f1"]
                    })

                self.logger.write(log_row)

                if self.save_best and self.is_best_metric(current_metric):
                    self.best_metric = current_metric
                    self.save_checkpoint(
                        epoch=epoch,
                        metrics={
                            "train": train_metrics,
                            "val": val_metrics
                        },
                        is_best=True
                    )

                    print(
                        f"Best {self.monitor} updated: "
                        f"{self.best_metric:.4f}"
                    )

                if self.scheduler is not None:
                    self.scheduler.step()
                    print(f"Current lr: {self.get_current_lr():.8f}")

                if self.early_stopping is not None:
                    should_stop = self.early_stopping.step(val_metrics)

                    print(
                        f"EarlyStopping counter: "
                        f"{self.early_stopping.counter}/{self.early_stopping.patience}"
                    )

                    if should_stop:
                        print(f"Early stopping triggered at epoch {epoch}.")
                        break
        print("Training finished.")

        self.test_best_model()
        print("Testing finished.")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

        return checkpoint

    @torch.no_grad()
    def test(self):
        if self.test_loader is None:
            print("No test loader provided.")
            return None

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        all_labels = []
        all_preds = []

        for batch in self.test_loader:
            batch = self.move_batch_to_device(batch)

            features = batch["features"]
            labels = batch["slide_label"].long()

            outputs = self.model(features)
            logits = outputs["logits"]

            loss = self.criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

        avg_loss = total_loss / total_samples
        metrics = compute_classification_metrics(all_labels, all_preds)
        metrics["loss"] = avg_loss

        print(
            f"Test Loss: {metrics['loss']:.4f} "
            f"Test Acc: {metrics['acc']:.4f} "
            f"Test Precision: {metrics['precision']:.4f} "
            f"Test Recall: {metrics['recall']:.4f} "
            f"Test Macro-F1: {metrics['macro_f1']:.4f}"
        )

        return metrics

    def test_best_model(self):
        if self.test_loader is None:
            print("No test loader provided, skip testing.")
            return None

        best_model_path = os.path.join(self.save_dir, "best_model.pth")

        if os.path.exists(best_model_path):
            checkpoint = self.load_checkpoint(best_model_path)
            test_metrics = self.test()
            self.save_test_results(test_metrics, checkpoint)
            return test_metrics

        else:
            print("Best model not found, use current model for testing.")
            test_metrics = self.test()
            self.save_test_results(test_metrics, checkpoint=None)
            return test_metrics

    def save_test_results(self, test_metrics, checkpoint=None):
        save_path = os.path.join(self.result_save_dir, self.result_filename)

        checkpoint_epoch = None
        if checkpoint is not None:
            checkpoint_epoch = checkpoint.get("epoch", None)

        row = {
            "dataset": self.cfg.get("dataset", ""),
            "task": self.cfg.get("task", ""),
            "fold": self.cfg.get("fold", ""),
            "model": self.cfg.get("model", {}).get("name", ""),
            "checkpoint_epoch": checkpoint_epoch,
            "test_loss": test_metrics.get("loss", None),
            "test_acc": test_metrics.get("acc", None),
            "test_precision": test_metrics.get("precision", None),
            "test_recall": test_metrics.get("recall", None),
            "test_macro_f1": test_metrics.get("macro_f1", None)
        }

        file_exists = os.path.exists(save_path)

        with open(save_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

        print(f"Test results saved to: {save_path}")
