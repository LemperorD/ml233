"""本作业训练、恢复和批量推理。"""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import FaceDataset, columns
from .models import FaceCNN, masked_mse


def device_for(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA 不可用')
    return torch.device(name)


def make_loader(root, frame, config, labeled=True, augment=False):
    return DataLoader(FaceDataset(root, frame, labeled, augment), batch_size=config['batch_size'],
                      shuffle=augment, num_workers=0, generator=torch.Generator().manual_seed(config['seed']))


def predict_values(artifact, root, frame, labeled=True, device='auto'):
    if artifact['columns'] != columns(root):
        raise ValueError('模型与数据的关键点顺序不一致')
    if artifact['kind'] == 'mean':
        return np.tile(artifact['mean'], (len(frame), 1))
    target = device_for(device)
    model = FaceCNN().to(target)
    model.load_state_dict(artifact['state'])
    model.eval()
    result = []
    with torch.no_grad():
        for images, _, _ in make_loader(root, frame, artifact['config'], labeled):
            result.append((model(images.to(target)).cpu().numpy() * 48 + 48))
    return np.concatenate(result)


def train_model(root, train, config, validation=None):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    names = columns(root)
    means = train[names].mean().to_numpy()
    if not np.isfinite(means).all():
        raise ValueError('训练折某坐标完全缺失，无法训练，请调整划分')
    if config['model'] == 'mean':
        return {'kind': 'mean', 'mean': means, 'columns': names}, [], 1
    target = device_for(config['device'])
    model = FaceCNN().to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    batches = make_loader(root, train, config, augment=True)
    valid_batches = make_loader(root, validation, config) if validation is not None else None
    best, best_epoch, best_score, stale, history = None, 1, float('inf'), 0, []
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total, count = 0.0, 0
        for images, targets, mask in batches:
            images, targets, mask = images.to(target), targets.to(target), mask.to(target)
            optimizer.zero_grad()
            predictions = model(images)
            loss = masked_mse(predictions, targets, mask)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * int(mask.sum())
            count += int(mask.sum())
        row = {'epoch': epoch, 'train_mse_normalized': total / max(1, count)}
        if valid_batches is not None:
            model.eval()
            squared, n = 0.0, 0
            with torch.no_grad():
                for images, targets, mask in valid_batches:
                    prediction = model(images.to(target)).cpu()
                    squared += float(((prediction - targets).square() * mask).sum()) * 48 ** 2
                    n += int(mask.sum())
            score = float(np.sqrt(squared / n))
            row['validation_rmse_pixels'] = score
            if score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
                best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        print(row, flush=True)
        if valid_batches is not None and stale >= config['patience']:
            break
    if valid_batches is not None:
        model.load_state_dict(best)
    else:
        best_epoch = config['epochs']
    return {'kind': 'cnn', 'state': model.cpu().state_dict(), 'columns': names, 'config': config.copy()}, history, best_epoch
