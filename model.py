import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
class NoticeboardModel(nn.Module):
    def __init__(self, num_labels=6):  # Changed from 5 to 6
        super().__init__()
        self.base = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels),  # Now outputs 6 values
            nn.Sigmoid()  # For multi-label classification
        )

    def forward(self, x):
        return self.base(x)
