"""评估保留的验证模型，保存分类指标、混淆矩阵及错分清单。"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import CLASSES, CLASS_TO_IDX, DEFAULT_DATA, fingerprint, image_groups, read_data
from .engine import predict_probabilities


def evaluate_run(run, root=DEFAULT_DATA, device='auto'):
    run = Path(run)
    frame = read_data(root)
    splits = json.loads((run / 'split.json').read_text())
    if fingerprint(frame, image_groups(root, frame)) != splits['fingerprint']:
        raise ValueError('训练 CSV 或图片内容已变化，不能复用旧划分')
    validation = frame.iloc[splits['validation_indices']]
    artifact = joblib.load(run / 'validation_model.joblib')
    probabilities = predict_probabilities(artifact, root, validation, device=device)
    y = validation.label.map(CLASS_TO_IDX).to_numpy()
    predictions = probabilities.argmax(1)
    labels = np.arange(len(CLASSES))
    result = {'accuracy': float(accuracy_score(y, predictions)),
              'macro_f1': float(f1_score(y, predictions, labels=labels, average='macro', zero_division=0)),
              'classification_report': classification_report(y, predictions, labels=labels,
                  target_names=CLASSES, output_dict=True, zero_division=0)}
    pd.DataFrame(confusion_matrix(y, predictions, labels=labels), index=CLASSES, columns=CLASSES).to_csv(run / 'confusion_matrix.csv')
    rows = validation[['id', 'label']].copy()
    rows['prediction'] = [CLASSES[i] for i in predictions]
    rows['confidence'] = probabilities.max(1)
    rows.to_csv(run / 'validation_predictions.csv', index=False)
    rows[rows.label != rows.prediction].to_csv(run / 'errors.csv', index=False)
    (run / 'evaluation.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads >= 1')
    import torch
    torch.set_num_threads(args.threads)
    print(json.dumps(evaluate_run(args.run_dir, args.data_dir, args.device), indent=2))


if __name__ == '__main__':
    main()
