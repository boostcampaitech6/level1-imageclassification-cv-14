import torch.nn as nn
from torchvision import models
from transformers import ViTForImageClassification
from base import BaseModel

class MultiTaskEfficientB0(BaseModel):
    def __init__(self, num_classes=8):
        super().__init__()
        self.pretrained_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.pretrained_model.classifier[1] = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        return self.pretrained_model(x)
    
class ViT(BaseModel):
    def __init__(self, num_classes=8):
        super().__init__()
        self.pretrained_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.pretrained_model.classifier = nn.Linear(self.pretrained_model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.pretrained_model(x)