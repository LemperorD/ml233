"""成对图块训练与整图恢复；验证按全像素累计误差。"""
import random
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import DataLoader
from .data import PatchDataset, read_image
from .models import SmallUNet


def device_for(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA 不可用')
    return torch.device(name)


def restore_image(artifact, image, device='auto', model=None):
    if artifact['kind'] == 'identity':
        return image.copy()
    if artifact['kind'] == 'median':
        source = Image.fromarray(np.rint(image * 255).astype(np.uint8))
        return np.asarray(source.filter(ImageFilter.MedianFilter(3)), dtype=np.float32) / 255
    if model is None:
        target = device_for(device)
        model = SmallUNet().to(target)
        model.load_state_dict(artifact['state'])
    target = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(image.copy()[None, None]).to(target)
        return model(inputs).cpu().numpy()[0, 0]


def predict_images(artifact, root, frame, labeled=False, device='auto'):
    model = None
    if artifact['kind'] == 'unet':
        model = SmallUNet().to(device_for(device))
        model.load_state_dict(artifact['state'])
        model.eval()
    return {row.image_id: restore_image(artifact, read_image(root, row.image_id, 'train_noisy' if labeled else 'test_noisy'), device, model)
            for row in frame.itertuples()}


def metrics(root, frame, predictions):
    squared, absolute, count = 0.0, 0.0, 0
    by_variant = {}
    for row in frame.itertuples():
        clean = read_image(root, row.image_id, 'train_clean')
        error = predictions[row.image_id].astype(np.float64) - clean
        sse, sae, pixels = float(np.square(error).sum()), float(np.abs(error).sum()), error.size
        squared += sse
        absolute += sae
        count += pixels
        totals = by_variant.setdefault(row.variant, [0.0, 0])
        totals[0] += sse
        totals[1] += pixels
    mse = squared / count
    return {'rmse': float(np.sqrt(mse)), 'mae': absolute / count,
            'psnr_db': float(-10 * np.log10(mse)) if mse > 0 else None,
            'pixels': count, 'by_variant': {name: {'rmse': float(np.sqrt(sse / n)), 'pixels': n}
                                          for name, (sse, n) in by_variant.items()}}


def train_model(root, train, config, validation=None):
    if config['model'] != 'unet':
        return {'kind': config['model']}, [], 1
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    target = device_for(config['device'])
    model = SmallUNet().to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    batches = DataLoader(PatchDataset(root, train, config['patch_size'], config['patches_per_image']),
                         batch_size=config['batch_size'], shuffle=True, num_workers=0,
                         generator=torch.Generator().manual_seed(config['seed']))
    best, best_epoch, best_score, stale, history = None, 1, float('inf'), 0, []
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total, pixels = 0.0, 0
        for noisy, clean in batches:
            noisy, clean = noisy.to(target), clean.to(target)
            optimizer.zero_grad()
            loss = (model(noisy) - clean).square().mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * clean.numel()
            pixels += clean.numel()
        row = {'epoch': epoch, 'train_mse': total / pixels}
        if validation is not None:
            artifact = {'kind': 'unet', 'state': model.state_dict()}
            # 复用当前网络，避免每页重新加载模型；整图大小保持不变。
            predictions = {item.image_id: restore_image(artifact, read_image(root, item.image_id, 'train_noisy'), config['device'], model)
                           for item in validation.itertuples()}
            score = metrics(root, validation, predictions)['rmse']
            row['validation_rmse'] = score
            if score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
                best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        print(row, flush=True)
        if validation is not None and stale >= config['patience']:
            break
    if validation is not None:
        model.load_state_dict(best)
    else:
        best_epoch = config['epochs']
    return {'kind': 'unet', 'state': model.cpu().state_dict(), 'config': config.copy()}, history, best_epoch
