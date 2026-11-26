"""作家风格识别：独立数据读取、划分及提交校验。"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DEFAULT_DATA = Path(__file__).resolve().parents[2] / 'datasets' / 'ml-hw2-group'
TARGET = 'author'
ID = 'id'
TEMPLATE = 'submission_example.csv'
CLASSES = ['EAP', 'HPL', 'MWS']
FEATURES = ['text']

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
    if frame.text.isna().any() or (frame.text.str.strip().str.len() == 0).any():
        raise ValueError('文本不能缺失或为空')
    return frame.text.astype(str)


def split_data(frame, seed=42):
    normalized = prepare(frame).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    groups = pd.factorize(normalized)[0]
    if len(np.unique(groups)) < len(frame):
        train, valid = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(frame, groups=groups))
    else:
        train, valid = train_test_split(np.arange(len(frame)), test_size=0.2, random_state=seed, stratify=frame.author)
    if set(frame.iloc[train].author) != set(CLASSES) or set(frame.iloc[valid].author) != set(CLASSES):
        raise ValueError('训练或验证未覆盖全部作者')
    return np.asarray(train), np.asarray(valid)


def write_submission(root, frame, values, output):
    template = pd.read_csv(Path(root) / TEMPLATE)
    values = np.asarray(values, dtype=float)
    if list(template.columns) != [ID, *CLASSES] or not template[ID].is_unique:
        raise ValueError('模板应为 id,EAP,HPL,MWS')
    if values.shape != (len(frame), len(CLASSES)) or not np.isfinite(values).all():
        raise ValueError('概率形状不符或含非有限值')
    if (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1):
        raise ValueError('概率必须在 [0,1] 且每行和为 1')
    predictions = pd.DataFrame(values, index=frame[ID], columns=CLASSES)
    if not predictions.index.is_unique or set(template[ID]) != set(predictions.index):
        raise ValueError('预测 ID 与模板不符')
    template[CLASSES] = predictions.loc[template[ID], CLASSES].to_numpy()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output, index=False)
