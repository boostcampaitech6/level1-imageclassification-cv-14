import timm
import torch.nn as nn
from torchvision import models
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
        self.pretrained_model = timm.create_model('vit_base_patch16_224',pretrained=True)
        self.pretrained_model.head = nn.Linear(self.pretrained_model.head.in_features, num_classes)

    def forward(self, x):
        return self.pretrained_model(x)
    
class DeiTModel(BaseModel):
    def __init__(self, num_classes=8, model_name='deit_base_distilled_patch16_224'):
        super().__init__()
        self.pretrained_model = timm.create_model(model_name, pretrained=True)

        in_features = self.pretrained_model.head.in_features
        self.pretrained_model.head = nn.Linear(in_features, num_classes)
        self.pretrained_model.head_dist = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.pretrained_model(x)

class Volo(BaseModel):
    def __init__(self, num_classes=8, model_name='volo_d2_224'):
        super().__init__()
        self.pretrained_model = timm.create_model(model_name, pretrained=True)

        # This is for main head
        in_features = self.pretrained_model.head.in_features
        self.pretrained_model.head = nn.Linear(in_features, num_classes)

        # This is for aux head
        aux_in_features = self.pretrained_model.aux_head.in_features
        self.pretrained_model.aux_head = nn.Linear(aux_in_features, num_classes)
    
    def forward(self, x):
        return self.pretrained_model(x)

class ResNestModel(BaseModel):
    def __init__(self, num_classes=8, model_name='resnest50d'):
        super().__init__()
        self.pretrained_model = timm.create_model(model_name, pretrained=True)

        in_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.pretrained_model(x)