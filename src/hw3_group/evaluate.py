"""有效坐标上的全局像素 RMSE 和逐坐标误差。"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from .data import DEFAULT_DATA, columns, read_data, image_groups, fingerprint, resolve_image
from .engine import predict_values


def metrics(values, predictions):
    mask = np.isfinite(values)
    squared = np.where(mask, (predictions - np.nan_to_num(values)) ** 2, 0)
    return {'rmse': float(np.sqrt(squared.sum() / mask.sum())), 'valid_coordinates': int(mask.sum())}


def evaluate_run(run, root=DEFAULT_DATA, device='auto'):
    run = Path(run)
    frame = read_data(root)
    split = json.loads((run / 'split.json').read_text())
    if fingerprint(frame, image_groups(root, frame)) != split['fingerprint']:
        raise ValueError('训练 CSV 或图片内容变化')
    validation = frame.iloc[split['validation_indices']]
    artifact = joblib.load(run / 'validation_model.joblib')
    predictions = predict_values(artifact, root, validation, device=device)
    names = columns(root)
    values = validation[names].to_numpy(dtype=float)
    result = metrics(values, predictions)
    result['by_coordinate'] = {name: metrics(values[:, i], predictions[:, i]) if np.isfinite(values[:, i]).any()
                               else {'rmse': None, 'valid_coordinates': 0} for i, name in enumerate(names)}
    complete = np.isfinite(values).all(axis=1)
    result['by_annotation'] = {key: metrics(values[mask], predictions[mask]) for key, mask in
                               [('complete', complete), ('sparse', ~complete)] if mask.any()}
    rows = pd.DataFrame(predictions, columns=names)
    rows.insert(0, 'ImageId', validation.ImageId.to_numpy())
    rows.to_csv(run / 'validation_predictions.csv', index=False)
    previews = run / 'previews'
    previews.mkdir(exist_ok=True)
    for i, row in enumerate(validation.head(6).itertuples()):
        with Image.open(resolve_image(root, row.image_path)) as source:
            image = source.convert('RGB')
        draw = ImageDraw.Draw(image)
        for point in range(0, len(names), 2):
            x, y = values[i, point:point + 2]
            if np.isfinite([x, y]).all():
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill='red')
            x, y = predictions[i, point:point + 2]
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill='lime')
        image.resize((384, 384)).save(previews / f'{row.ImageId}.png')

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
