import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(True),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogleNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)  # Changed from 7 to 4
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
      #  print(f'After pre_layers: {x.shape}')
        x = self.inception3a(x)
      #  print(f'After inception3a: {x.shape}')
        x = self.inception3b(x)
      #  print(f'After inception3b: {x.shape}')
        x = self.maxpool(x)
      #  print(f'After maxpool: {x.shape}')
        x = self.inception4a(x)
      #  print(f'After inception4a: {x.shape}')
        x = self.inception4b(x)
      #  print(f'After inception4b: {x.shape}')
        x = self.inception4c(x)
      #  print(f'After inception4c: {x.shape}')
        x = self.inception4d(x)
      #  print(f'After inception4d: {x.shape}')
        x = self.inception4e(x)
      #  print(f'After inception4e: {x.shape}')
        x = self.maxpool(x)
      #  print(f'After second maxpool: {x.shape}')
        x = self.inception5a(x)
      #  print(f'After inception5a: {x.shape}')
        x = self.inception5b(x)
      #  print(f'After inception5b: {x.shape}')
        x = self.avgpool(x)
      #  print(f'After avgpool: {x.shape}')
        x = x.view(x.size(0), -1)
      #  print(f'After view: {x.shape}')
        x = self.dropout(x)
        x = self.linear(x)
        return x

def test():
    net = GoogleNet()
    y = net(Variable(torch.randn(1, 3, 32, 32)))  # Adjusted input size to 32x32
    print(y.size())

test()
