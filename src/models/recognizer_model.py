import torch
import torch.nn as nn
from collections import OrderedDict

class Conv3DBlock(nn.Module):
    """Basic 3D convolutional block with Conv3D, BatchNorm3D, and ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv3DBlock, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)),
            ('bn', nn.BatchNorm3d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """Residual block for X3D."""
    def __init__(self, in_channels, channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channels, channels, kernel_size=1, stride=1, bias=False)),
            ('bn', nn.BatchNorm3d(channels))
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(channels , channels, kernel_size=3, stride=1, padding=1, bias=False, groups=channels)),
            ('bn', nn.BatchNorm3d(channels))
        ]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(channels, out_channels, kernel_size=1, stride=1, bias=False)),
            ('bn', nn.BatchNorm3d(out_channels))
        ]))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(OrderedDict([
                ('conv', nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(1,1,1), bias=False)),
                ('bn', nn.BatchNorm3d(out_channels))
            ]))

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        print(out.shape, identity.shape)
        out += identity
        out = self.relu(out)
        return out

class Backbone(nn.Module):
    """Backbone for X3D-s."""
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1_s = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(17, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)),
        ]))

        self.conv1_t = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(24, 24, kernel_size=(5, 1, 1), stride=(1, 2, 2), padding=(2, 0, 0), bias=False,groups=24)), #spatial stride 2
            ('bn', nn.BatchNorm3d(24)),
        ]))

        self.layer1 = self._make_layer(24, 54, 24, 2, downsample=True)
        self.layer2 = self._make_layer(24, 108, 48, 5, downsample=True)
        self.layer3 = self._make_layer(48, 216, 96, 3, downsample=True)

        self.conv5 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(96, 216, kernel_size=1, stride=1, bias=False)),
            ('bn', nn.BatchNorm3d(216)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def _make_layer(self, in_channels, channels, out_channels, blocks, downsample):
        layers = [ResidualBlock(in_channels, channels, out_channels, downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        x = self.conv1_s(x)
        #print(x.shape)
        x = self.conv1_t(x)
        #print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv5(x)
        return x


class Recognizer3D(nn.Module):
    def __init__(self, num_classes=99):
        super(Recognizer3D, self).__init__()
        self.backbone = Backbone()
        self.cls_head = nn.Sequential(
            OrderedDict([
                ('avgpool', nn.AdaptiveAvgPool3d((1, 1, 1))),
                ('flatten', nn.Flatten()),
                ('fc_cls', nn.Linear(216, num_classes))
            ])
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.cls_head(x)
        return x