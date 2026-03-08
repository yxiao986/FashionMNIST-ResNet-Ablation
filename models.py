import torch
import torch.nn as nn
import torch.nn.functional as F

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_residual=True):
        super(BasicBlock, self).__init__()
        self.use_residual = use_residual
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if self.use_residual and (stride != 1 or in_channels != out_channels):
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], 
                                (0, 0, 0, 0, 0, out_channels - in_channels), 
                                "constant", 0)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_residual:
            out += self.shortcut(x)
            
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, n, num_classes=10, use_residual=True, input_channels=1):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, n, stride=1)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, self.use_residual))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def plain_20(num_classes=10, input_channels=1):
    return ResNet(BasicBlock, 
                  n=3, 
                  num_classes=num_classes, 
                  use_residual=False, 
                  input_channels=input_channels)

def plain_44(num_classes=10, input_channels=1):
    return ResNet(BasicBlock, 
                  n=7, 
                  num_classes=num_classes, 
                  use_residual=False, 
                  input_channels=input_channels)

def resnet_20(num_classes=10, input_channels=1):
    return ResNet(BasicBlock, 
                  n=3, 
                  num_classes=num_classes, 
                  use_residual=True, 
                  input_channels=input_channels)

def resnet_56(num_classes=10, input_channels=1):
    return ResNet(BasicBlock, 
                  n=9,
                  num_classes=num_classes, 
                  use_residual=True, 
                  input_channels=input_channels)

def resnet_44(num_classes=10, input_channels=1):
    return ResNet(BasicBlock, 
                  n=7, 
                  num_classes=num_classes, 
                  use_residual=True, 
                  input_channels=input_channels)

