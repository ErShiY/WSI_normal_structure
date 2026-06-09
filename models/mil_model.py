import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_classes=2, dropout=0.25):
        super().__init__()

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        """
        features: [B, N, D]
        """

        if features.dim() == 2:
            features = features.unsqueeze(0)

        h = self.feature_proj(features)          # [B, N, hidden_dim]
        attn = self.attention(h)                # [B, N, 1]
        attn = torch.softmax(attn, dim=1)       # [B, N, 1]

        slide_feat = torch.sum(attn * h, dim=1) # [B, hidden_dim]
        logits = self.classifier(slide_feat)    # [B, num_classes]

        return {
            "logits": logits,
            "attention": attn,
            "slide_feat": slide_feat
        }

