import timm
import torch.nn as nn
from base import BaseModel
from torchvision import models

class ResNet34(BaseModel):
    def __init__(self, num_classes=18):
        super().__init__()
        self.pretrained_model = models.resnet34(pretrained=True)
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        self.pretrained_model.fc = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        return self.pretrained_model(x)
    
class EfficientNetB0(BaseModel):
    def __init__(self, num_classes=18):
        super().__init__()
        self.pretrained_model = models.efficientnet_b0(pretrained=models.EfficientNet_B0_Weights)
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        num_ftrs = self.pretrained_model.classifier[1].in_features
        self.pretrained_model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.pretrained_model(x)
    
class ViTcls18(BaseModel):
    def __init__(self, num_classes=18):
        super().__init__()
        self.pretrained_model = timm.create_model('vit_base_patch16_224',pretrained=True)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.pretrained_model.head = nn.Linear(self.pretrained_model.head.in_features, num_classes)

    def forward(self, x):
        return self.pretrained_model(x)
    
class ViTcls3(BaseModel):
    def __init__(self, num_classes=3):
        super().__init__()
        self.pretrained_model = timm.create_model('vit_base_patch16_224',pretrained=True)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.pretrained_model.head = nn.Linear(self.pretrained_model.head.in_features, num_classes)

    def forward(self, x):
        return self.pretrained_model(x)
    
class ViTcls2(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        self.pretrained_model = timm.create_model('vit_base_patch16_224',pretrained=True)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.pretrained_model.head = nn.Linear(self.pretrained_model.head.in_features, num_classes)

    def forward(self, x):
        return self.pretrained_model(x)