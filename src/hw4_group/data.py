"""文档成对图像、文档分组及流式像素提交。"""
from pathlib import Path
import hashlib
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit

DEFAULT_DATA = Path(__file__).resolve().parents[2] / 'datasets' / 'ml-hw4-group'


def read_image(root, image_id, folder):
    if not isinstance(image_id, str) or Path(image_id).name != image_id:
        raise ValueError('非法文档 ID')
    with Image.open(Path(root) / folder / f'{image_id}.png') as source:
        return np.asarray(source.convert('L'), dtype=np.float32) / 255


def read_data(root=DEFAULT_DATA, labeled=True):
    frame = pd.read_csv(Path(root) / ('train_manifest.csv' if labeled else 'test_manifest.csv'))
    if not {'image_id', 'height', 'width', 'variant'}.issubset(frame.columns):
        raise ValueError('manifest 缺少必要字段')
    if frame.image_id.isna().any() or not frame.image_id.is_unique or not frame.variant.isin(['original', 'mixed']).all():
        raise ValueError('文档 ID 或 variant 不合法')
    for row in frame.itertuples():
        if not row.image_id.endswith('_' + row.variant):
            raise ValueError('文档 ID 与 variant 不一致')
        for folder in (['train_noisy', 'train_clean'] if labeled else ['test_noisy']):
            image = read_image(root, row.image_id, folder)
            if image.shape != (row.height, row.width) or min(image.shape) < 1:
                raise ValueError(f'{row.image_id} 实际尺寸与 manifest 不一致')
    return frame


def document_groups(frame):
    return frame.image_id.str.replace(r'_(original|mixed)$', '', regex=True).to_numpy()


def split_data(frame, seed=42):
    groups = document_groups(frame)
    if len(np.unique(groups)) < 2:
        raise ValueError('至少需要两份独立文档进行验证')
    return next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(frame, groups=groups))


def fingerprint(root, frame):
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    for row in frame.itertuples():
        for folder in ['train_noisy', 'train_clean']:
            digest.update(read_image(root, row.image_id, folder).tobytes())
    return digest.hexdigest()


class PatchDataset(Dataset):
    def __init__(self, root, frame, patch_size=128, patches_per_image=4):
        self.root, self.frame = Path(root), frame.reset_index(drop=True)
        self.patch_size, self.patches_per_image = patch_size, patches_per_image

    def __len__(self):
        return len(self.frame) * self.patches_per_image

    def __getitem__(self, index):
        row = self.frame.iloc[index // self.patches_per_image]
        noisy = read_image(self.root, row.image_id, 'train_noisy')
        clean = read_image(self.root, row.image_id, 'train_clean')
        pad = ((0, max(0, self.patch_size - noisy.shape[0])), (0, max(0, self.patch_size - noisy.shape[1])))
        noisy, clean = np.pad(noisy, pad, mode='edge'), np.pad(clean, pad, mode='edge')
        top = random.randint(0, noisy.shape[0] - self.patch_size)
        left = random.randint(0, noisy.shape[1] - self.patch_size)
        region = np.s_[top:top + self.patch_size, left:left + self.patch_size]
        return torch.from_numpy(noisy[region].copy()[None]), torch.from_numpy(clean[region].copy()[None])


def write_submission(root, frame, predictions, output, chunksize=100000):
    if chunksize < 1 or set(predictions) != set(frame.image_id):
        raise ValueError('分块大小或预测图 ID 不合法')
    for row in frame.itertuples():
        values = predictions[row.image_id]
        if values.shape != (row.height, row.width) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError('预测图尺寸错误或灰度超出 [0,1]')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.part')
    seen = {name: np.zeros(values.size, dtype=bool) for name, values in predictions.items()}
    count, first = 0, True
    try:
        with temporary.open('w', newline='') as destination, pd.read_csv(
                Path(root) / 'sample_submission.csv', chunksize=chunksize, dtype={'id': str}) as reader:
            for chunk in reader:
                if list(chunk.columns) != ['id', 'value']:
                    raise ValueError('模板必须为 id,value')
                parts = chunk.id.str.rsplit('_', n=2, expand=True)
                if parts.shape[1] != 3 or parts.isna().any().any():
                    raise ValueError('像素 ID 格式错误')
                positions = np.arange(len(chunk))
                values = np.empty(len(chunk), dtype=np.float32)
                for name in parts[0].unique():
                    if name not in predictions:
                        raise ValueError(f'模板出现未知图片：{name}')
                    selected = (parts[0] == name).to_numpy()
                    row_text, col_text = parts.loc[selected, 1], parts.loc[selected, 2]
                    if not row_text.str.fullmatch(r'[1-9][0-9]*').all() or not col_text.str.fullmatch(r'[1-9][0-9]*').all():
                        raise ValueError('像素坐标必须为 1 基正整数')
                    rows, cols = row_text.to_numpy(dtype=int) - 1, col_text.to_numpy(dtype=int) - 1
                    image = predictions[name]
                    if (rows >= image.shape[0]).any() or (cols >= image.shape[1]).any():
                        raise ValueError('模板像素坐标越界')
                    flat = rows * image.shape[1] + cols
                    if len(np.unique(flat)) != len(flat) or seen[name][flat].any():
                        raise ValueError('模板含重复像素')
                    seen[name][flat] = True
                    values[positions[selected]] = image[rows, cols]
                chunk['value'] = values
                chunk.to_csv(destination, index=False, header=first)
                count += len(chunk)
                first = False
        if count != sum(values.size for values in predictions.values()) or not all(values.all() for values in seen.values()):
            raise ValueError('模板未覆盖全部测试像素')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return count
