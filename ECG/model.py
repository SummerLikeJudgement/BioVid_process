from torch import nn
import torch

class CNN1D(nn.Module):
    def __init__(self, num_classes=5, input_length=2816):
        super(CNN1D, self).__init__()
        self.input_length = input_length
        self.feature = nn.Sequential(
            # 第一层: kernel=7, padding=3, stride=1 → 长度不变: 2816
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # → 1408

            # 第二层: kernel=5, padding=2, stride=1 → 长度不变: 1408
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # → 704

            # 第三层: kernel=3, padding=1, stride=1 → 长度不变: 704
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # → 352

            # 第四层: kernel=3, padding=1, stride=1 → 长度不变: 352
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # → 176

            # 第五层: 进一步提取特征
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化 → 1
        )

        self._calculate_fc_input_dim()

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.fc_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def _calculate_fc_input_dim(self):
        """计算全连接层的输入维度"""
        # 模拟前向传播计算特征维度
        x = torch.randn(1, 1, self.input_length)
        with torch.no_grad():
            x = self.feature_extractor(x)
        self.fc_input_dim = x.shape[1]  # 512

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

