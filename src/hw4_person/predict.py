"""加载全量模型并按提交模板 ID 顺序导出。"""
import argparse
from pathlib import Path
import joblib
from .data import DEFAULT_DATA, read_data, prepare, write_submission
from .models import predict_values


def predict_run(run, root=DEFAULT_DATA, output=None):
    run = Path(run)
    frame = read_data(root, labeled=False)
    model = joblib.load(run / 'model.joblib')
    output = Path(output) if output else run / 'submission.csv'
    write_submission(root, frame, predict_values(model, prepare(frame)), output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(predict_run(args.run_dir, args.data_dir, args.output))


if __name__ == '__main__':
    main()
