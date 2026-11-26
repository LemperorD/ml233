"""关键点分组验证、掩码训练、全量重训和提交。"""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import torch
from .data import DEFAULT_DATA, columns, read_data, image_groups, split_data, fingerprint
from .engine import train_model
from .evaluate import evaluate_run, metrics
from .predict import predict_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/hw3_group/baseline'))
    parser.add_argument('--model', choices=['mean', 'cnn'], default='cnn')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--skip-predict', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error('轮数、批量、耐心轮数、线程与学习率必须为正')
    torch.set_num_threads(args.threads)
    run = args.output_dir
    run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()):
        raise FileExistsError('实验目录必须为空')
    started = time.perf_counter()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    frame = read_data(args.data_dir)
    groups = image_groups(args.data_dir, frame)
    train_idx, valid_idx = split_data(args.data_dir, frame, groups, args.seed)
    split = {'fingerprint': fingerprint(frame, groups), 'train_indices': train_idx.tolist(), 'validation_indices': valid_idx.tolist(),
             'train_ids': frame.iloc[train_idx].ImageId.tolist(), 'validation_ids': frame.iloc[valid_idx].ImageId.tolist()}
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    (run / 'split.json').write_text(json.dumps(split, indent=2))
    model, history, best_epoch = train_model(args.data_dir, frame.iloc[train_idx], config, frame.iloc[valid_idx])
    joblib.dump(model, run / 'validation_model.joblib')
    result = evaluate_run(run, args.data_dir, args.device)
    names = columns(args.data_dir)
    baseline = np.tile(frame.iloc[train_idx][names].mean().to_numpy(), (len(valid_idx), 1))
    result['mean_baseline'] = metrics(frame.iloc[valid_idx][names].to_numpy(), baseline)
    final, final_history, _ = train_model(args.data_dir, frame, dict(config, epochs=best_epoch))
    joblib.dump(final, run / 'model.joblib')
    if not args.skip_predict:
        predict_run(run, args.data_dir, device=args.device)
    result.update(best_epoch=best_epoch, seconds=time.perf_counter() - started)
    (run / 'metrics.json').write_text(json.dumps(result, indent=2))
    (run / 'history.json').write_text(json.dumps({'validation_training': history, 'full_training': final_history}, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
