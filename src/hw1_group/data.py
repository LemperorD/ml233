"""共享单车租赁需求预测：独立数据读取、划分及提交校验。"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DEFAULT_DATA = Path(__file__).resolve().parents[2] / 'datasets' / 'ml-hw1-group'
TARGET = 'count'
ID = 'datetime'
TEMPLATE = 'submission_example.csv'
CLASSES = []
FEATURES = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
CATEGORICAL = ['season', 'holiday', 'workingday', 'weather']
NUMERIC = ['temp', 'atemp', 'humidity', 'windspeed', 'year', 'month', 'hour', 'weekday', 'dayofyear', 'elapsed_days', 'hour_sin', 'hour_cos']

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
    dates = pd.to_datetime(frame[ID], errors='raise')
    result = frame[FEATURES].copy()
    result['year'] = dates.dt.year
    result['month'] = dates.dt.month
    result['hour'] = dates.dt.hour
    result['weekday'] = dates.dt.dayofweek
    result['dayofyear'] = dates.dt.dayofyear
    result['elapsed_days'] = (dates - pd.Timestamp('2011-01-01')).dt.total_seconds() / 86400
    result['hour_sin'] = np.sin(2 * np.pi * dates.dt.hour / 24)
    result['hour_cos'] = np.cos(2 * np.pi * dates.dt.hour / 24)
    for column in CATEGORICAL:
        result[column] = result[column].fillna('__MISSING__').astype(str)
    return result


def split_data(frame, seed=42):
    day = pd.to_datetime(frame[ID]).dt.day
    train = np.flatnonzero(day <= 11)
    valid = np.flatnonzero((day >= 12) & (day <= 15))
    if len(train) == 0 or len(valid) == 0:
        raise ValueError('需要每月 1–11 日训练和 12–15 日验证数据')
    if (frame[TARGET] < 0).any():
        raise ValueError('租车需求不能为负')
    return train, valid


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
