"""课程个人作业 2：图片读取、训练增强和提交对齐。"""
from pathlib import Path
import hashlib
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset

DEFAULT_DATA = Path(__file__).resolve().parents[2] / 'datasets' / 'ml-hw1-person'
CLASSES = ['burger', 'churros', 'crepes', 'falafel', 'hot_dog', 'pad_thai',
           'pani_puri', 'pretzel', 'shawarma', 'tacos']
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASSES)}


def read_data(root=DEFAULT_DATA, labeled=True):
    root = Path(root)
    frame = pd.read_csv(root / ('train.csv' if labeled else 'test.csv'))
    required = ['id', 'label'] if labeled else ['id']
    if not set(required).issubset(frame.columns):
        raise ValueError(f'CSV 必须包含 {required}')
    if frame.id.isna().any() or not frame.id.is_unique:
        raise ValueError('图片 id 必须非空且唯一')
    if labeled and not frame.label.isin(CLASSES).all():
        raise ValueError('训练标签包含未知类别或空值')
    folder = root / ('train_images' if labeled else 'test_images')
    for name in frame.id:
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f'非法图片文件名：{name}')
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)
    return frame


def image_groups(root, frame):
    # 解码后的 RGB 像素哈希可识别元信息不同但像素完全相同的图片。
    hashes = []
    for name in frame.id:
        with Image.open(Path(root) / 'train_images' / name) as image:
            rgb = image.convert('RGB')
            hashes.append(hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest())
    return np.asarray(hashes)


def fingerprint(frame, groups):
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(''.join(groups).encode())
    return digest.hexdigest()


class FoodDataset(Dataset):
    def __init__(self, root, frame, labeled=True, augment=False, image_size=128):
        self.folder = Path(root) / ('train_images' if labeled else 'test_images')
        self.frame = frame.reset_index(drop=True)
        self.labeled = labeled
        self.augment = augment
        self.image_size = image_size

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.folder / row.id) as source:
            image = source.convert('RGB')
        if self.augment:
            # 仅训练时裁剪/翻转/颜色扰动；验证整图缩放并填充，保留食物主体。
            scale = random.uniform(0.85, 1.0)
            width, height = image.size
            crop_w, crop_h = max(1, int(width * scale)), max(1, int(height * scale))
            left, top = random.randint(0, width - crop_w), random.randint(0, height - crop_h)
            image = image.crop((left, top, left + crop_w, top + crop_h))
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))
        image = ImageOps.pad(image, (self.image_size, self.image_size), method=Image.Resampling.BILINEAR)
        values = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(values.transpose(2, 0, 1).copy())
        # 无预训练网络使用固定缩放，不从验证集估计归一化参数。
        tensor = (tensor - 0.5) / 0.5
        target = CLASS_TO_IDX[row.label] if self.labeled else -1
        return tensor, target


def histogram_features(root, frame, labeled=True):
    folder = Path(root) / ('train_images' if labeled else 'test_images')
    result = []
    for name in frame.id:
        with Image.open(folder / name) as source:
            values = np.asarray(source.convert('RGB').resize((128, 128)))
        features = np.concatenate([np.histogram(values[:, :, c], bins=16, range=(0, 256))[0]
                                   for c in range(3)]).astype(float)
        result.append(features / (128 * 128))
    return np.asarray(result)


def write_submission(root, ids, predictions, output):
    template = pd.read_csv(Path(root) / 'submission_example.csv')
    if list(template.columns) != ['id', 'label'] or not template.id.is_unique:
        raise ValueError('模板必须是唯一 id,label 两列')
    if len(ids) != len(predictions) or not set(predictions).issubset(CLASSES):
        raise ValueError('预测长度或标签不合法')
    result = pd.Series(predictions, index=ids)
    if not result.index.is_unique or set(result.index) != set(template.id):
        raise ValueError('预测 ID 与模板不一致')
    template['label'] = template.id.map(result)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output, index=False)
