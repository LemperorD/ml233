"""独立加载保留的验证模型并重算指标，保存逐样本预测。"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, mean_absolute_error, mean_squared_error
from .data import DEFAULT_DATA, TARGET, ID, CLASSES, read_data, prepare, fingerprint
from .models import predict_values

def metrics(y, predictions):
    return {'rmsle': float(np.sqrt(np.mean((np.log1p(y) - np.log1p(predictions)) ** 2))),
            'rmse': float(np.sqrt(mean_squared_error(y, predictions)))}


def evaluate_run(run, root=DEFAULT_DATA):
    run = Path(run)
    frame = read_data(root)
    split = json.loads((run / 'split.json').read_text())
    if fingerprint(frame) != split['fingerprint']:
        raise ValueError('训练数据内容或顺序变化，拒绝使用历史划分')
    validation = frame.iloc[split['validation_indices']]
    model = joblib.load(run / 'validation_model.joblib')
    predictions = predict_values(model, prepare(validation))
    result = metrics(validation[TARGET], predictions)
    rows = validation[[ID, TARGET]].copy()
    rows['prediction'] = predictions
    rows['residual'] = rows.prediction - rows[TARGET]
    rows.to_csv(run / 'validation_predictions.csv', index=False)
    (run / 'evaluation.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    print(json.dumps(evaluate_run(args.run_dir, args.data_dir), indent=2))


if __name__ == '__main__':
    main()
