"""葡萄酒分组交叉验证、全量重训和提交生成。"""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from .data import DEFAULT_DATA, FEATURES, feature_groups, fingerprint, read_data
from .evaluate import evaluate_run, metrics
from .models import build_model
from .predict import predict_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/hw1_person/ridge'))
    parser.add_argument('--model', choices=['ridge', 'extra_trees', 'mlp'], default='ridge')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--trees', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.folds < 2 or args.epochs < 1 or args.trees < 1 or args.alpha < 0:
        parser.error('folds >= 2，epochs/trees >= 1，alpha >= 0')
    run = args.output_dir
    run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()):
        raise FileExistsError(f'实验目录非空，请使用新的 --output-dir：{run}')
    started = time.perf_counter()
    frame = read_data(args.data_dir)
    groups = feature_groups(frame)
    if args.folds > len(np.unique(groups)):
        parser.error('折数不能超过特征分组数')
    (run / 'config.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(args).items()}, indent=2))
    splits, scores, baseline = [], [], np.zeros(len(frame))
    for fold, (train_idx, valid_idx) in enumerate(GroupKFold(args.folds).split(frame, groups=groups)):
        model = build_model(args.model, args.seed, args.alpha, args.trees, args.epochs)
        model.fit(frame.iloc[train_idx][FEATURES], frame.iloc[train_idx].quality)
        prediction = model.predict(frame.iloc[valid_idx][FEATURES])
        score = metrics(frame.iloc[valid_idx].quality, prediction)
        baseline[valid_idx] = frame.iloc[train_idx].quality.mean()
        splits.append({'train_indices': train_idx.tolist(), 'validation_indices': valid_idx.tolist(),
                       'train_ids': frame.iloc[train_idx].id.tolist(),
                       'validation_ids': frame.iloc[valid_idx].id.tolist()})
        scores.append(score)
        joblib.dump(model, run / f'fold_{fold}.joblib')
        print(f'fold {fold + 1}: {score}', flush=True)
    (run / 'splits.json').write_text(json.dumps({'fingerprint': fingerprint(frame), 'folds': splits}))
    result = evaluate_run(run, args.data_dir)
    model = build_model(args.model, args.seed, args.alpha, args.trees, args.epochs)
    model.fit(frame[FEATURES], frame.quality)
    joblib.dump(model, run / 'model.joblib')
    predict_run(run, args.data_dir)
    result.update(folds=scores, mean_baseline=metrics(frame.quality, baseline),
                  duplicate_rows=int(frame.duplicated(FEATURES, keep=False).sum()),
                  conflicting_groups=int((frame.groupby(FEATURES).quality.nunique() > 1).sum()),
                  seconds=time.perf_counter() - started)
    (run / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
