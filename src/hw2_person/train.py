"""美食分层分组验证、早停、全量重训与提交生成；不下载预训练权重。"""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
import torch

from .data import CLASSES, DEFAULT_DATA, fingerprint, image_groups, read_data
from .engine import train_model
from .evaluate import evaluate_run
from .predict import predict_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/hw2_person/baseline'))
    parser.add_argument('--model', choices=['cnn', 'histogram'], default='cnn')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--folds', type=int, default=5, help='使用第 1 折验证；默认约 20%%')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.image_size < 32 or args.folds < 2 or args.lr <= 0:
        parser.error('epochs/patience/batch-size/threads >= 1，image-size >= 32，folds >= 2，lr > 0')
    torch.set_num_threads(args.threads)
    run = args.output_dir
    run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()):
        raise FileExistsError(f'实验目录非空，请使用新的 --output-dir：{run}')
    started = time.perf_counter()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    frame = read_data(args.data_dir)
    groups = image_groups(args.data_dir, frame)
    if any(len(set(groups[frame.label == label])) < args.folds for label in CLASSES):
        raise ValueError('每类独立图片组数必须不少于 folds')
    if len(np.unique(groups)) == len(frame):
        # 无重复时直接分层抽样，避免分组启发式在小数据上遗漏类别。
        train_idx, valid_idx = train_test_split(np.arange(len(frame)), test_size=1 / args.folds,
                                                stratify=frame.label, random_state=args.seed)
        split_method = "stratified_holdout"
    else:
        splitter = StratifiedGroupKFold(args.folds, shuffle=True, random_state=args.seed)
        train_idx, valid_idx = next(splitter.split(frame, frame.label, groups))
        split_method = "stratified_group_fold"
    if set(groups[train_idx]) & set(groups[valid_idx]):
        raise RuntimeError('重复图片组跨越训练与验证')
    if set(frame.iloc[train_idx].label) != set(CLASSES) or set(frame.iloc[valid_idx].label) != set(CLASSES):
        raise ValueError('划分未覆盖全部类别，请调整 seed 或 folds')
    split = {'method': split_method, 'fingerprint': fingerprint(frame, groups), 'train_indices': train_idx.tolist(),
             'validation_indices': valid_idx.tolist(), 'train_ids': frame.iloc[train_idx].id.tolist(),
             'validation_ids': frame.iloc[valid_idx].id.tolist(),
             'duplicate_rows': int(len(frame) - len(np.unique(groups)))}
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    (run / 'class_to_idx.json').write_text(json.dumps({name: i for i, name in enumerate(CLASSES)}, indent=2))
    (run / 'split.json').write_text(json.dumps(split, indent=2))
    artifact, history, best_epoch = train_model(args.data_dir, frame.iloc[train_idx], config, frame.iloc[valid_idx])
    joblib.dump(artifact, run / 'validation_model.joblib')
    result = evaluate_run(run, args.data_dir, args.device)
    # 验证模型永久保留；全量模型仅用于测试集，避免把全量训练分数当验证分数。
    final_config = dict(config, epochs=best_epoch)
    final, final_history, _ = train_model(args.data_dir, frame, final_config)
    joblib.dump(final, run / 'model.joblib')
    predict_run(run, args.data_dir, device=args.device)
    result.update(best_epoch=best_epoch, majority_baseline=float(
        (frame.iloc[valid_idx].label == frame.iloc[train_idx].label.mode().iloc[0]).mean()),
        seconds=time.perf_counter() - started)
    (run / 'history.json').write_text(json.dumps({'validation_training': history, 'full_training': final_history}, indent=2))
    (run / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'classification_report'}, indent=2))


if __name__ == '__main__':
    main()
