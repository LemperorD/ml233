"""按原文档分组、图块训练、整图验证、全量重训。"""
import argparse
import json
from pathlib import Path
import time
import joblib
import torch
from .data import DEFAULT_DATA, read_data, split_data, fingerprint, read_image
from .engine import train_model, metrics
from .evaluate import evaluate_run
from .predict import predict_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/hw4_group/baseline'))
    parser.add_argument('--model', choices=['identity', 'median', 'unet'], default='unet')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--patches-per-image', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--export-submission', action='store_true', help='训练后导出约 1130 万行 CSV')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.patch_size, args.patches_per_image, args.threads) < 1 or args.lr <= 0:
        parser.error('训练参数必须为正')
    torch.set_num_threads(args.threads)
    run = args.output_dir
    run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()):
        raise FileExistsError('实验目录必须为空')
    started = time.perf_counter()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    frame = read_data(args.data_dir)
    train_idx, valid_idx = split_data(frame, args.seed)
    split = {'fingerprint': fingerprint(args.data_dir, frame), 'train_indices': train_idx.tolist(), 'validation_indices': valid_idx.tolist(),
             'train_ids': frame.iloc[train_idx].image_id.tolist(), 'validation_ids': frame.iloc[valid_idx].image_id.tolist()}
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    (run / 'split.json').write_text(json.dumps(split, indent=2))
    model, history, best_epoch = train_model(args.data_dir, frame.iloc[train_idx], config, frame.iloc[valid_idx])
    joblib.dump(model, run / 'validation_model.joblib')
    result = evaluate_run(run, args.data_dir, args.device)
    validation = frame.iloc[valid_idx]
    result['identity_baseline'] = metrics(args.data_dir, validation, {row.image_id: read_image(args.data_dir, row.image_id, 'train_noisy') for row in validation.itertuples()})
    final, final_history, _ = train_model(args.data_dir, frame, dict(config, epochs=best_epoch))
    joblib.dump(final, run / 'model.joblib')
    if args.export_submission:
        predict_run(run, args.data_dir, device=args.device)
    result.update(best_epoch=best_epoch, seconds=time.perf_counter() - started)
    (run / 'metrics.json').write_text(json.dumps(result, indent=2))
    (run / 'history.json').write_text(json.dumps({'validation_training': history, 'full_training': final_history}, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
