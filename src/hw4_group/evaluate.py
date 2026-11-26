"""独立整图评估与恢复、误差预览。"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
import torch
from .data import DEFAULT_DATA, read_data, read_image, fingerprint
from .engine import predict_images, metrics


def evaluate_run(run, root=DEFAULT_DATA, device='auto'):
    run = Path(run)
    frame = read_data(root)
    split = json.loads((run / 'split.json').read_text())
    if fingerprint(root, frame) != split['fingerprint']:
        raise ValueError('训练 manifest 或成对图片已变化')
    validation = frame.iloc[split['validation_indices']]
    artifact = joblib.load(run / 'validation_model.joblib')
    predictions = predict_images(artifact, root, validation, labeled=True, device=device)
    result = metrics(root, validation, predictions)
    previews = run / 'previews'
    previews.mkdir(exist_ok=True)
    for row in validation.head(4).itertuples():
        noisy = read_image(root, row.image_id, 'train_noisy')
        clean = read_image(root, row.image_id, 'train_clean')
        prediction = predictions[row.image_id]
        error = np.clip(np.abs(prediction - clean) * 4, 0, 1)
        panels = np.concatenate([noisy, prediction, clean, error], axis=1)
        Image.fromarray(np.rint(panels * 255).astype(np.uint8)).save(previews / f'{row.image_id}.png')
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
    torch.set_num_threads(args.threads)
    print(json.dumps(evaluate_run(args.run_dir, args.data_dir, args.device), indent=2))


if __name__ == '__main__':
    main()
