"""固定验证、全量重训与实验结果保存。"""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from .data import DEFAULT_DATA, TARGET, ID, CLASSES, read_data, prepare, split_data, fingerprint
from .models import build_model, training_target
from .evaluate import evaluate_run, metrics
from .predict import predict_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/hw2_group/baseline'))
    parser.add_argument('--model', choices=['logistic', 'naive_bayes', 'mlp'], default='logistic')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--trees', type=int, default=100)
    parser.add_argument('--features', choices=['word', 'char', 'both'], default='word')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--skip-predict', action='store_true', help='只训练评估，不导出测试预测')
    args = parser.parse_args()
    if min(args.epochs, args.trees, args.threads) < 1:
        parser.error('epochs/trees/threads >= 1')
    run = args.output_dir
    run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()):
        raise FileExistsError('实验目录必须为空，请更换 --output-dir')
    started = time.perf_counter()
    frame = read_data(args.data_dir)
    train_idx, valid_idx = split_data(frame, args.seed)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    split = {'fingerprint': fingerprint(frame), 'train_indices': train_idx.tolist(), 'validation_indices': valid_idx.tolist(),
             'train_ids': frame.iloc[train_idx][ID].tolist(), 'validation_ids': frame.iloc[valid_idx][ID].tolist()}
    (run / 'split.json').write_text(json.dumps(split, indent=2))
    history = {}
    with threadpool_limits(limits=args.threads):
        model = build_model(args.model, args.seed, args.epochs, args.trees, args.features)
        print('训练验证阶段模型', flush=True)
        model.fit(prepare(frame.iloc[train_idx]), training_target(frame.iloc[train_idx][TARGET]))
        history['validation_training'] = list(getattr(model[-1], 'loss_curve_', []))
        joblib.dump(model, run / 'validation_model.joblib')
        result = evaluate_run(run, args.data_dir)
        prior = frame.iloc[train_idx][TARGET].value_counts(normalize=True).reindex(CLASSES).to_numpy()
        result['baseline'] = metrics(frame.iloc[valid_idx][TARGET], np.tile(prior, (len(valid_idx), 1)))
        print('验证结果：', result, flush=True)
        final = build_model(args.model, args.seed, args.epochs, args.trees, args.features)
        print('全量重训', flush=True)
        final.fit(prepare(frame), training_target(frame[TARGET]))
        history['full_training'] = list(getattr(final[-1], 'loss_curve_', []))
        joblib.dump(final, run / 'model.joblib')
        if not args.skip_predict:
            predict_run(run, args.data_dir)
    result['seconds'] = time.perf_counter() - started
    (run / 'history.json').write_text(json.dumps(history, indent=2))
    (run / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
