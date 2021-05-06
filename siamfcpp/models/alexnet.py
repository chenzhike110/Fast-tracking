import torch
import torch.nn as nn
from .common_block import conv_bn_relu

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0, bias=True, groups=1),
                                    nn.BatchNorm2d(96),
                                    nn.ReLU())
        self.pool1 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=0, bias=True, groups=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        self.pool2 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=0, bias=True, groups=1),
                                    nn.BatchNorm2d(384),
                                    nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0, bias=True, groups=1),
                                    nn.BatchNorm2d(384),
                                    nn.ReLU())
        self.conv5 = conv_bn_relu(384, 256, 1, 3, 0, has_relu=False)
        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=0, bias=True, groups=1),
                                    nn.BatchNorm2d(256))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def update_params(self, path=None):
        if path != None:
            try:
                state_dict = torch.load(
                    path,
                    map_location=torch.device("cuda"))
            except:
                state_dict = torch.load(
                    path,
                    map_location=torch.device("cpu"))
            self.load_state_dict(state_dict, strict=False)