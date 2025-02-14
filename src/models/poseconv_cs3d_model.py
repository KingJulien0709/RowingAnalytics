import torch
import torch.nn as nn
from collections import OrderedDict

class Backbone(nn.Module):
    def __init__(self, in_channels):
        super(Backbone, self).__init__()
        self.conv1a = nn.Sequential(
            OrderedDict([
                ('conv',nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1),bias=False)),
                ('bn', nn.BatchNorm3d(32))
            ])
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2a = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1),bias=False)),
                ('bn', nn.BatchNorm3d(64))
            ])
        )
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1),bias=False)),
                ('bn', nn.BatchNorm3d(128))
            ])
        )
        self.conv3b = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1),bias=False)),
                ('bn', nn.BatchNorm3d(128))
            ])
        )
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4a = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1),bias=False)),
                ('bn', nn.BatchNorm3d(256))
            ])
        )
        self.conv4b = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1),bias=False)),
                ('bn', nn.BatchNorm3d(256))
            ])
        )
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        x = self.pool1(self.conv1a(x))
        x = self.pool2(self.conv2a(x))
        x = self.pool3(self.conv3b(self.conv3a(x)))
        x = self.pool4(self.conv4b(self.conv4a(x)))
        return x


class PoseC3Ds(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(PoseC3Ds, self).__init__()
        self.backbone = Backbone(in_channels)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.cls_head = nn.Sequential(
            OrderedDict([
                ('fc_cls', nn.Linear(256, num_classes))
            ])
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.cls_head(x)
        return x