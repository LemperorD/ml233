"""重新加载各折模型，评估未参与该折训练的样本。"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data import DEFAULT_DATA, FEATURES, fingerprint, read_data


def metrics(y, predictions):
    return {'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
            'mae': float(mean_absolute_error(y, predictions))}


def evaluate_run(run, root=DEFAULT_DATA):
    run = Path(run)
    frame = read_data(root)
    splits = json.loads((run / 'splits.json').read_text())
    if fingerprint(frame) != splits['fingerprint']:
        raise ValueError('训练 CSV 内容或顺序已变化，不能复用旧划分')
    predictions = np.full(len(frame), np.nan)
    for fold, split in enumerate(splits['folds']):
        indices = split['validation_indices']
        model = joblib.load(run / f'fold_{fold}.joblib')
        predictions[indices] = model.predict(frame.iloc[indices][FEATURES])
    if not np.isfinite(predictions).all():
        raise ValueError('折外预测未覆盖所有样本')
    result = metrics(frame.quality, predictions)
    rows = frame[['id', 'quality']].copy()
    rows['prediction'] = predictions
    rows['residual'] = predictions - frame.quality
    rows.to_csv(run / 'oof_predictions.csv', index=False)
    by_quality = {str(q): metrics(part.quality, part.prediction)
                  for q, part in rows.groupby('quality')}
    result['by_quality'] = by_quality
    (run / 'evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    print(json.dumps(evaluate_run(args.run_dir, args.data_dir), indent=2))


if __name__ == '__main__':
    main()
