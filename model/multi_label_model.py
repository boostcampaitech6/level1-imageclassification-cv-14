import torch.nn as nn
from torchvision import models
from base import BaseModel

class MultiLabelEfficientB0(BaseModel):
    def __init__(self, num_classes=18):
        super().__init__()
        self.pretrained_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.pretrained_model.classifier[1] = nn.Linear(1280, 8)
    
    def forward(self, x):
        return self.pretrained_model(x)