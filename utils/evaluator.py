import torch
from utils.metrics import compute_classification_metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        outputs = model(batch)
        loss = criterion(outputs, batch)

        logits = outputs["logits"]
        labels = batch["label"]

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        total_loss += loss.item()

    metrics = compute_classification_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader)

    return metrics


def move_batch_to_device(batch, device):
    new_batch = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            new_batch[key] = value.to(device)
        else:
            new_batch[key] = value

    return new_batch