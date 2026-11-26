"""本作业内复用的回归模型和 MLP 网络。"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class WineMLP(nn.Module):
    def __init__(self, input_dim=11, width=32):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(),
                                    nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class MLPRegressor(RegressorMixin, BaseEstimator):
    """可放入 sklearn Pipeline；每折只拟合训练折的标准化与目标均值。"""
    def __init__(self, width=32, epochs=100, lr=1e-3, batch_size=64, seed=42):
        self.width = width
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, x, y):
        torch.manual_seed(self.seed)
        self.model_ = WineMLP(x.shape[1], self.width)
        self.target_mean_ = float(np.mean(y))
        dataset = TensorDataset(torch.tensor(np.asarray(x), dtype=torch.float32),
                                torch.tensor(np.asarray(y) - self.target_mean_, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed))
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-3)
        self.loss_history_ = []
        for _ in range(self.epochs):
            self.model_.train()
            loss_sum = 0.0
            for inputs, targets in loader:
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.model_(inputs), targets)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(inputs)
            self.loss_history_.append(loss_sum / len(dataset))
        return self

    def predict(self, x):
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.tensor(np.asarray(x), dtype=torch.float32)).numpy() + self.target_mean_


def build_model(name, seed=42, alpha=10.0, trees=300, epochs=100):
    if name == 'ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if name == 'extra_trees':
        return ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=2,
                                   max_depth=12, random_state=seed, n_jobs=-1)
    if name == 'mlp':
        return make_pipeline(StandardScaler(), MLPRegressor(epochs=epochs, seed=seed))
    raise ValueError(f'未知模型：{name}')
