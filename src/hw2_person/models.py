"""仅供美食作业复用的小型 CNN 与传统特征分类器。"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


class FoodCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        blocks = []
        channels = [3, 16, 32, 64, 128]
        for inputs, outputs in zip(channels[:-1], channels[1:]):
            blocks.extend([nn.Conv2d(inputs, outputs, 3, padding=1),
                           nn.BatchNorm2d(outputs), nn.ReLU(inplace=True), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name, num_classes=10, seed=42):
    if name == 'cnn':
        return FoodCNN(num_classes)
    if name == 'histogram':
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
    raise ValueError(f'未知模型：{name}')
