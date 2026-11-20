import torch
import torch.nn as nn
import torch.nn.functional as F


class Base_Model(nn.Module):
    def __init__(self):
        super(Base_Model, self).__init__()

        self.backbone = None
        self.output = None

    def forward(self, x):
        features = self.backbone(x)
        logits = self.output(features)
        return logits

    def get_backbone(self, x):
        return self.backbone(x)

    def backbone_frozen_forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        logits = self.output(features)
        return logits, features

    def get_backbone_dim(self):
        return self.output.in_features


class CNN_MNIST(Base_Model):
    def __init__(self):
        super(CNN_MNIST, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Linear(512, 10)


class CNN_FMNIST(Base_Model):
    def __init__(self):
        super(CNN_FMNIST, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Linear(512, 10)


class CNN_CIFAR_10(Base_Model):
    def __init__(self):
        super(CNN_CIFAR_10, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Linear(192, 10)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=16, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class ResNet_CIFAR100(Base_Model):
    def __init__(self, num_classes=100):
        super(ResNet_CIFAR100, self).__init__()
        self.in_channels = 64
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            self._make_layer(BasicBlock, 64, 2, stride=1),
            self._make_layer(BasicBlock, 128, 2, stride=2),
            self._make_layer(BasicBlock, 256, 2, stride=2),
            self._make_layer(BasicBlock, 512, 2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.output = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=16, num_channels=out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


class ResNet_CINIC10(Base_Model):
    def __init__(self, num_classes=10):
        super(ResNet_CINIC10, self).__init__()
        self.in_channels = 64
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            self._make_layer(BasicBlock, 64, 2, stride=1),
            self._make_layer(BasicBlock, 128, 2, stride=2),
            self._make_layer(BasicBlock, 256, 2, stride=2),
            self._make_layer(BasicBlock, 512, 2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.output = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=16, num_channels=out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(16, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(16, out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.gn3 = nn.GroupNorm(16, out_channels * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = F.relu(self.gn1(self.conv1(x)))
        out = F.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class ResNet_TinyImageNet(Base_Model):
    def __init__(self, num_classes=200):
        super(ResNet_TinyImageNet, self).__init__()
        self.in_channels = 64

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),

            self._make_layer(Bottleneck, 64, 3, stride=1),  # 64x64
            self._make_layer(Bottleneck, 128, 4, stride=2),  # 32x32
            self._make_layer(Bottleneck, 256, 6, stride=2),  # 16x16
            self._make_layer(Bottleneck, 512, 3, stride=2),  # 8x8

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.output = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        out_dim = out_channels * block.expansion

        if stride != 1 or self.in_channels != out_dim:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_dim,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(16, out_dim),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_dim

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
