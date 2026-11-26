"""使用全量训练的模型预测测试图片。"""
import argparse
from pathlib import Path
import joblib
from .data import CLASSES, DEFAULT_DATA, read_data, write_submission
from .engine import predict_probabilities


def predict_run(run, root=DEFAULT_DATA, output=None, device='auto'):
    run = Path(run)
    frame = read_data(root, labeled=False)
    artifact = joblib.load(run / 'model.joblib')
    probabilities = predict_probabilities(artifact, root, frame, labeled=False, device=device)
    output = Path(output) if output else run / 'submission.csv'
    write_submission(root, frame.id, [CLASSES[i] for i in probabilities.argmax(1)], output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads >= 1')
    import torch
    torch.set_num_threads(args.threads)
    print(predict_run(args.run_dir, args.data_dir, args.output, args.device))


if __name__ == '__main__':
    main()
