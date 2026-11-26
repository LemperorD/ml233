"""背包价格预测：独立数据读取、划分及提交校验。"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DEFAULT_DATA = Path(__file__).resolve().parents[2] / 'datasets' / 'ml-hw3-person'
TARGET = 'Price'
ID = 'id'
TEMPLATE = 'sample_submission.csv'
CLASSES = []
NUMERIC = ['Compartments', 'Weight Capacity (kg)']
CATEGORICAL = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
FEATURES = NUMERIC + CATEGORICAL

def read_data(root=DEFAULT_DATA, labeled=True):
    frame = pd.read_csv(Path(root) / ('train.csv' if labeled else 'test.csv'))
    required = [ID, *FEATURES] + ([TARGET] if labeled else [])
    if not set(required).issubset(frame.columns):
        raise ValueError(f'缺少字段：{set(required) - set(frame.columns)}')
    if frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError('样本 ID 必须非空且唯一')
    if labeled:
        if frame[TARGET].isna().any():
            raise ValueError('标签不能缺失')
        if CLASSES and not frame[TARGET].isin(CLASSES).all():
            raise ValueError('标签包含未知类别')
        if not CLASSES and not np.isfinite(frame[TARGET].to_numpy(dtype=float)).all():
            raise ValueError('目标值必须有限')
    return frame


def fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()

def prepare(frame):
    result = frame[FEATURES].copy()
    for column in CATEGORICAL:
        result[column] = result[column].fillna('__MISSING__').astype(str)
    if not np.isfinite(result[NUMERIC].to_numpy(dtype=float)[~result[NUMERIC].isna().to_numpy()]).all():
        raise ValueError('数值特征含无穷值')
    return result


def split_data(frame, seed=42):
    features = prepare(frame)
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    if len(np.unique(groups)) != len(frame):
        train, valid = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(frame, groups=groups))
    else:
        train, valid = train_test_split(np.arange(len(frame)), test_size=0.2, random_state=seed,
                                       stratify=frame[TARGET] if CLASSES else None)
    if CLASSES and (set(frame.iloc[train][TARGET]) != set(CLASSES) or set(frame.iloc[valid][TARGET]) != set(CLASSES)):
        raise ValueError('训练或验证未覆盖全部类别，请调整 seed 或增加数据')
    return np.asarray(train), np.asarray(valid)


def write_submission(root, frame, values, output):
    template = pd.read_csv(Path(root) / TEMPLATE)
    if list(template.columns) != [ID, TARGET] or not template[ID].is_unique:
        raise ValueError('提交模板字段或 ID 不合法')
    values = np.asarray(values)
    if len(values) != len(frame):
        raise ValueError('预测长度不符')
    if CLASSES:
        if not set(values).issubset(CLASSES):
            raise ValueError('未知预测标签')
    elif not np.isfinite(values.astype(float)).all():
        raise ValueError('预测必须有限')
    predictions = pd.Series(values, index=frame[ID])
    if not predictions.index.is_unique or set(template[ID]) != set(predictions.index):
        raise ValueError('预测 ID 与模板不符')
    template[TARGET] = template[ID].map(predictions)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output, index=False)
