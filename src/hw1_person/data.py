"""课程个人作业 1：葡萄酒数据及提交格式。"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

DEFAULT_DATA = Path(__file__).resolve().parents[2] / 'datasets' / 'ml-hw1-person'
FEATURES = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']


def read_data(root=DEFAULT_DATA, labeled=True):
    frame = pd.read_csv(Path(root) / ('train.csv' if labeled else 'test.csv'))
    required = ['id', *FEATURES] + (['quality'] if labeled else [])
    if not set(required).issubset(frame.columns):
        raise ValueError(f'缺少字段：{set(required) - set(frame.columns)}')
    if frame.id.isna().any() or not frame.id.is_unique:
        raise ValueError('id 必须非空且唯一')
    values = frame[FEATURES + (['quality'] if labeled else [])].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('特征或标签含空值/无穷值')
    return frame


def feature_groups(frame):
    # 使用完整特征元组，保证重复记录不会跨折；不删除冲突标签。
    return pd.factorize(pd.MultiIndex.from_frame(frame[FEATURES]))[0]


def fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def write_submission(root, ids, predictions, output):
    template = pd.read_csv(Path(root) / 'submission_example.csv')
    predictions = np.asarray(predictions, dtype=float)
    if list(template.columns) != ['id', 'quality'] or not template.id.is_unique:
        raise ValueError('提交模板应为唯一 id,quality 两列')
    if len(predictions) != len(ids) or not np.isfinite(predictions).all():
        raise ValueError('预测长度不符或含非有限值')
    result = pd.Series(predictions, index=ids)
    if not result.index.is_unique or set(template.id) != set(result.index):
        raise ValueError('预测 ID 与模板不一致')
    template['quality'] = template.id.map(result)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output, index=False)
