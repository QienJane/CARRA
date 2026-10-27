import torch
from torch import nn
from torch.nn import functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Contrastive temperature must be positive.")
        self.temperature = float(temperature)

    def forward(self, features, labels):
        if features.ndim != 3 or features.shape[1] < 2:
            raise ValueError("Expected contrastive features shaped batch,views,dimension with views >= 2.")
        batch, views, _ = features.shape
        labels = labels.reshape(-1).to(features.device)
        if batch == 0 or labels.numel() != batch:
            raise ValueError("One label is required for every nonempty sample.")
        
        features = F.normalize(features.reshape(batch * views, -1), dim=1)
        labels = labels.repeat_interleave(views)
        logits = features @ features.T / self.temperature
        diagonal = torch.eye(batch * views, dtype=torch.bool, device=features.device)
        positives = labels[:, None].eq(labels[None, :]) & ~diagonal
        log_denominator = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
        log_probability = logits - log_denominator[:, None]
        positive_counts = positives.sum(dim=1)
        return -((log_probability * positives).sum(dim=1) / positive_counts).mean()
