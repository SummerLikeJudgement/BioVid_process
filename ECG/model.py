from torch import nn
import torch

class BasicBlock1D(nn.Module):
    """1D ResNet 基本残差块：Conv-BN-ReLU-Conv-BN + 残差短接 + ReLU"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    轻量 1D ResNet，用于时序信号二分类。
    结构：stem → layer1 → layer2 → layer3 → layer4 → GAP → classifier
    - 支持任意输入长度（AdaptiveAvgPool1d）
    - 默认为二分类（num_classes=2）
    - 可通过 width_mul 控制通道规模
    """
    def __init__(self, num_classes=2, in_channels=1, base_planes=32, blocks=(2, 2, 2, 2),
                 kernel_size=7, width_mul=1.0, dropout=0.3):
        super().__init__()
        b = lambda c: int(c * width_mul)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, b(base_planes), kernel_size=kernel_size, stride=2,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(b(base_planes)),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Stages
        self.in_planes = b(base_planes)
        self.layer1 = self._make_layer(BasicBlock1D, b(base_planes), blocks[0], stride=1, kernel_size=kernel_size)
        self.layer2 = self._make_layer(BasicBlock1D, b(base_planes*2), blocks[1], stride=2, kernel_size=kernel_size)
        self.layer3 = self._make_layer(BasicBlock1D, b(base_planes*4), blocks[2], stride=2, kernel_size=kernel_size)
        self.layer4 = self._make_layer(BasicBlock1D, b(base_planes*8), blocks[3], stride=2, kernel_size=kernel_size)

        # Head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(b(base_planes*8), num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=7):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride=stride, downsample=downsample, kernel_size=kernel_size))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, downsample=None, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 1, T)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).squeeze(-1)  # (B, C)
        x = self.dropout(x)
        x = self.fc(x)               # (B, num_classes)
        return x


# 为了兼容你现有的导入方式（from ECG.model import CNN1D），
# 提供一个别名类，内部实际使用 ResNet1D。
class CNN1D(ResNet1D):
    def __init__(self, num_classes=2, input_length=None):
        # input_length 在 ResNet 中不需要（自适应池化），保留该参数以避免改动训练代码
        super().__init__(num_classes=num_classes, in_channels=1, base_planes=32,
                         blocks=(2, 2, 2, 2), kernel_size=7, width_mul=1.0, dropout=0.3)