import torch


pt = torch.load(f"outputs/checkpoints/best_model.pth")
print(pt)