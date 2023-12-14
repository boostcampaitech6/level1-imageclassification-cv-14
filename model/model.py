import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=True):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels, out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)]

        if norm:
          layers+=[nn.BatchNorm2d(out_channels)]
        if relu:
          layers+=[nn.ReLU(inplace=False)] 
        self.conv_block=nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

class CNN(BaseModel):
    def __init__(self, num_classes=18):
        super().__init__()
        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1) 
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(64, 256, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=False)
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x) 
        x = self.conv2(x) 
        x = self.dropout(x) 
        x = self.max_pool(x) 
        x = self.conv3(x) 
        x = self.dropout(x)
        x = self.avg_pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
