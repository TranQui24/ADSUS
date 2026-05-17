import torch
import torch.nn as nn
import torchvision.models as models

class HybridResNet50(nn.Module):
    def __init__(self, in_channels=5, num_classes=1):
        super(HybridResNet50, self).__init__()
        
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        
        old_weight = self.resnet.conv1.weight.data
        
      
        self.resnet.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
      
        self.resnet.conv1.weight.data[:, :3] = old_weight
        nn.init.kaiming_normal_(self.resnet.conv1.weight.data[:, 3:], mode='fan_out', nonlinearity='relu')
        
      
        self.resnet.fc = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.resnet(x)
        x = self.sigmoid(x) 
        return x