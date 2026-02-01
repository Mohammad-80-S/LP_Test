import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PSRLoss(nn.Module):
    """Combined pixel and contrastive loss for super resolution."""
    
    def __init__(self, alpha=1.0, beta=1.0, margin=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.siamese_network = models.resnet18(pretrained=True)
        self.siamese_network.fc = nn.Linear(
            self.siamese_network.fc.in_features, 128
        )

    def forward(self, sr_patches, hr_patches):
        pixel_loss = F.mse_loss(sr_patches, hr_patches)
        sr_embeddings = self.siamese_network(sr_patches)
        hr_embeddings = self.siamese_network(hr_patches)
        distance = torch.norm(sr_embeddings - hr_embeddings, p=2, dim=1)
        contrastive_loss = torch.mean(
            torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        )
        return self.alpha * pixel_loss + self.beta * contrastive_loss