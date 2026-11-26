"""人脸关键点图片、有效标签掩码和查表提交。"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import random
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DEFAULT_DATA = Path(__file__).resolve().parents[2] / 'datasets' / 'ml-hw3-group'


def resolve_image(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('图片路径必须为数据目录内的相对路径')
    candidates = [Path(root) / relative, Path(root) / relative.parts[0] / relative]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(candidates[0])


def columns(root):
    points = pd.read_csv(Path(root) / 'keypoints.csv')
    result = [column for row in points.itertuples() for column in (row.x_column, row.y_column)]
    if len(result) != 30 or len(set(result)) != 30:
        raise ValueError('关键点映射必须为 15 对、30 个唯一字段')
    return result


def read_data(root=DEFAULT_DATA, labeled=True):
    frame = pd.read_csv(Path(root) / ('train.csv' if labeled else 'test.csv'))
    required = ['ImageId', 'image_path'] + (columns(root) if labeled else [])
    if not set(required).issubset(frame.columns):
        raise ValueError('缺少图片 ID、路径或坐标字段')
    if frame.ImageId.isna().any() or not frame.ImageId.is_unique:
        raise ValueError('ImageId 必须唯一且非空')
    for relative in frame.image_path:
        resolve_image(root, relative)
    if labeled:
        values = frame[columns(root)].to_numpy(dtype=float)
        if np.isinf(values).any() or not np.isfinite(values).any(axis=1).all():
            raise ValueError('坐标不能为无穷值，每张图至少需要一个有效坐标')
    return frame


def image_groups(root, frame):
    result = []
    for relative in frame.image_path:
        with Image.open(resolve_image(root, relative)) as image:
            gray = image.convert('L')
            if gray.size != (96, 96):
                raise ValueError('人脸图片必须为 96×96')
            result.append(hashlib.sha256(gray.tobytes()).hexdigest())
    return np.asarray(result)


def fingerprint(frame, groups):
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(''.join(groups).encode())
    return digest.hexdigest()


def split_data(root, frame, groups, seed):
    if len(np.unique(groups)) < len(frame):
        return next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(frame, groups=groups))
    complete = frame[columns(root)].notna().all(axis=1)
    stratify = complete if complete.value_counts().min() >= 2 else None
    return train_test_split(np.arange(len(frame)), test_size=0.2, random_state=seed, stratify=stratify)


class FaceDataset(Dataset):
    def __init__(self, root, frame, labeled=True, augment=False):
        self.root, self.frame = Path(root), frame.reset_index(drop=True)
        self.labeled, self.augment = labeled, augment
        self.columns = columns(root)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(resolve_image(self.root, row.image_path)) as source:
            image = source.convert('L')
        if image.size != (96, 96):
            raise ValueError('人脸图片必须为 96×96')
        if self.augment:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        inputs = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()[None] / 255)
        values = row[self.columns].to_numpy(dtype=np.float32) if self.labeled else np.full(30, np.nan, dtype=np.float32)
        mask = np.isfinite(values)
        targets = np.where(mask, (values - 48) / 48, 0).astype(np.float32)
        return inputs, torch.from_numpy(targets), torch.from_numpy(mask)


def write_submission(root, frame, predictions, output):
    names = columns(root)
    predictions = np.asarray(predictions)
    if predictions.shape != (len(frame), 30) or not np.isfinite(predictions).all():
        raise ValueError('预测必须为有限的 N×30 像素坐标')
    lookup = pd.read_csv(Path(root) / 'IdLookupTable.csv')
    template = pd.read_csv(Path(root) / 'sample_submission.csv')
    if list(template.columns) != ['RowId', 'Location'] or not template.RowId.is_unique or not lookup.RowId.is_unique:
        raise ValueError('提交模板或查找表 RowId 不合法')
    images = pd.Index(frame.ImageId).get_indexer(lookup.ImageId)
    features = pd.Index(names).get_indexer(lookup.FeatureName)
    if (images < 0).any() or (features < 0).any() or set(template.RowId) != set(lookup.RowId):
        raise ValueError('查找表含未知图片/坐标，或 RowId 与模板不符')
    values = pd.Series(predictions[images, features], index=lookup.RowId)
    template['Location'] = template.RowId.map(values)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output, index=False)
