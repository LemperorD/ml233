"""训练、评估及预测共用的本作业计算过程。"""
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CLASSES, CLASS_TO_IDX, FoodDataset, histogram_features
from .models import build_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA 不可用，请选择 --device cpu')
    return torch.device(name)


def loader(root, frame, config, labeled=True, augment=False):
    return DataLoader(FoodDataset(root, frame, labeled, augment, config['image_size']),
                      batch_size=config['batch_size'], shuffle=augment, num_workers=0,
                      generator=torch.Generator().manual_seed(config['seed']))


def cnn_probabilities(model, batches, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for images, _ in batches:
            parts.append(model(images.to(device)).softmax(dim=1).cpu().numpy())
    return np.concatenate(parts)


def train_model(root, train, config, validation=None):
    seed_everything(config['seed'])
    if config['model'] == 'histogram':
        model = build_model('histogram', seed=config['seed'])
        model.fit(histogram_features(root, train), train.label.map(CLASS_TO_IDX))
        return {'kind': 'histogram', 'model': model, 'classes': CLASSES}, [], 1
    device = resolve_device(config['device'])
    model = build_model('cnn', len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-3)
    batches = loader(root, train, config, augment=True)
    valid_batches = loader(root, validation, config) if validation is not None else None
    best_state, best_accuracy, best_epoch, stale, history = None, -1.0, 0, 0, []
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        for images, targets in batches:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)
            correct += int((logits.argmax(1) == targets).sum())
        row = {'epoch': epoch, 'train_loss': loss_sum / len(train), 'train_accuracy': correct / len(train)}
        if valid_batches is not None:
            probabilities = cnn_probabilities(model, valid_batches, device)
            accuracy = float(np.mean(probabilities.argmax(1) == validation.label.map(CLASS_TO_IDX).to_numpy()))
            row['validation_accuracy'] = accuracy
            if accuracy > best_accuracy:
                best_accuracy, best_epoch, stale = accuracy, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        print(row, flush=True)
        if valid_batches is not None and stale >= config['patience']:
            break
    if valid_batches is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = config['epochs']
    artifact = {'kind': 'cnn', 'state': copy.deepcopy(model.cpu().state_dict()),
                'classes': CLASSES, 'config': config.copy()}
    return artifact, history, best_epoch


def predict_probabilities(artifact, root, frame, labeled=True, device='auto'):
    if artifact['classes'] != CLASSES:
        raise ValueError('模型类别顺序与当前代码不一致')
    if artifact['kind'] == 'histogram':
        model = artifact['model']
        probabilities = model.predict_proba(histogram_features(root, frame, labeled))
        result = np.zeros((len(frame), len(CLASSES)))
        result[:, model.classes_.astype(int)] = probabilities
        return result
    config = artifact['config']
    target = resolve_device(device)
    model = build_model('cnn', len(CLASSES)).to(target)
    model.load_state_dict(artifact['state'])
    return cnn_probabilities(model, loader(root, frame, config, labeled), target)
