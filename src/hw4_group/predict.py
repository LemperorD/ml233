"""整图恢复、保存浮点预测并流式导出全部测试像素。"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import torch
from .data import DEFAULT_DATA, read_data, write_submission
from .engine import predict_images


def predict_run(run, root=DEFAULT_DATA, output=None, device='auto', export_csv=True, chunksize=100000):
    run = Path(run)
    frame = read_data(root, labeled=False)
    artifact = joblib.load(run / 'model.joblib')
    predictions = predict_images(artifact, root, frame, device=device)
    directory = run / 'test_predictions'
    directory.mkdir(exist_ok=True)
    for name, image in predictions.items():
        np.save(directory / f'{name}.npy', image)
    if not export_csv:
        return directory
    output = Path(output) if output else run / 'submission.csv'
    count = write_submission(root, frame, predictions, output, chunksize)
    print(f'导出 {count} 个像素', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--images-only', action='store_true', help='仅保存 .npy，不导出大 CSV')
    parser.add_argument('--chunksize', type=int, default=100000)
    args = parser.parse_args()
    if args.threads < 1 or args.chunksize < 1:
        parser.error('threads/chunksize >= 1')
    torch.set_num_threads(args.threads)
    print(predict_run(args.run_dir, args.data_dir, args.output, args.device, not args.images_only, args.chunksize))


if __name__ == '__main__':
    main()
