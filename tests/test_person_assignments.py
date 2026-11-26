"""独立数据、分组隔离、模型保存及 CLI 端到端回归检查。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.hw1_person import data as wine
from src.hw1_person.evaluate import evaluate_run as evaluate_wine
from src.hw2_person import data as food
from src.hw2_person.evaluate import evaluate_run as evaluate_food
from src.hw2_person.models import FoodCNN
from src.hw1_person.models import WineMLP

PROJECT = Path(__file__).resolve().parents[1]


def run_cli(module, *args):
    result = subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=PROJECT,
                            capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)


class AssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.wine_root = cls.root / 'wine'
        cls.wine_root.mkdir()
        rng = np.random.default_rng(42)
        x = rng.normal(size=(30, 11))
        x[1] = x[0]  # 完全相同特征，标签可不同。
        train = pd.DataFrame(x, columns=wine.FEATURES)
        train.insert(0, 'id', np.arange(30))
        train['quality'] = 5 + x[:, 0] * 0.2
        train.to_csv(cls.wine_root / 'train.csv', index=False)
        test = train.drop(columns='quality').iloc[:4].copy()
        test['id'] = np.arange(100, 104)
        test.to_csv(cls.wine_root / 'test.csv', index=False)
        pd.DataFrame({'id': test.id[::-1], 'quality': 0.0}).to_csv(cls.wine_root / 'submission_example.csv', index=False)
        cls.food_root = cls.root / 'food'
        cls.food_root.mkdir()
        for folder in ['train_images', 'test_images']:
            (cls.food_root / folder).mkdir()
        rows = []
        for c, label in enumerate(food.CLASSES):
            for n in range(6):
                name = f'{c}_{n}.jpg'
                values = rng.integers(0, 256, (36, 48, 3), dtype=np.uint8)
                Image.fromarray(values).save(cls.food_root / 'train_images' / name)
                rows.append({'id': name, 'label': label})
        pd.DataFrame(rows).to_csv(cls.food_root / 'train.csv', index=False)
        test_ids = ['test_a.jpg', 'test_b.jpg', 'test_c.jpg']
        for name in test_ids:
            Image.new('RGB', (40, 50), (100, 120, 180)).save(cls.food_root / 'test_images' / name)
        pd.DataFrame({'id': test_ids}).to_csv(cls.food_root / 'test.csv', index=False)
        pd.DataFrame({'id': test_ids[::-1], 'label': 'burger'}).to_csv(cls.food_root / 'submission_example.csv', index=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_network_shapes_and_gradients(self):
        for model, inputs, shape in [(WineMLP(), torch.randn(3, 11), (3,)),
                                     (FoodCNN(), torch.randn(3, 3, 32, 32), (3, 10))]:
            output = model(inputs)
            self.assertEqual(tuple(output.shape), shape)
            output.sum().backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_wine_cli_models_grouping_scaling_and_reload(self):
        for model in ['ridge', 'extra_trees', 'mlp']:
            with self.subTest(model=model):
                run = self.root / f'wine_{model}'
                run_cli('src.hw1_person.train', '--data-dir', self.wine_root, '--output-dir', run,
                        '--model', model, '--folds', 3, '--epochs', 2, '--trees', 5)
                result = json.loads((run / 'metrics.json').read_text())
                again = evaluate_wine(run, self.wine_root)
                self.assertAlmostEqual(result['rmse'], again['rmse'])
                frame = wine.read_data(self.wine_root)
                groups = wine.feature_groups(frame)
                splits = json.loads((run / 'splits.json').read_text())['folds']
                seen = []
                for fold, split in enumerate(splits):
                    train_idx, valid_idx = split['train_indices'], split['validation_indices']
                    self.assertFalse(set(groups[train_idx]) & set(groups[valid_idx]))
                    seen.extend(valid_idx)
                    if model != 'extra_trees':
                        scaler = joblib.load(run / f'fold_{fold}.joblib')[0]
                        np.testing.assert_allclose(scaler.mean_, frame.iloc[train_idx][wine.FEATURES].mean())
                self.assertEqual(sorted(seen), list(range(len(frame))))
                run_cli('src.hw1_person.evaluate', '--data-dir', self.wine_root, '--run-dir', run)
                output = run / 'reload.csv'
                run_cli('src.hw1_person.predict', '--data-dir', self.wine_root, '--run-dir', run, '--output', output)
                pd.testing.assert_frame_equal(pd.read_csv(output), pd.read_csv(run / 'submission.csv'))
                self.assertEqual(pd.read_csv(output).id.tolist(), [103, 102, 101, 100])

    def test_food_cli_models_reload_and_validation_isolation(self):
        for model in ['histogram', 'cnn']:
            with self.subTest(model=model):
                run = self.root / f'food_{model}'
                run_cli('src.hw2_person.train', '--data-dir', self.food_root, '--output-dir', run,
                        '--model', model, '--epochs', 1, '--folds', 2, '--image-size', 32,
                        '--batch-size', 16, '--device', 'cpu', '--threads', 1)
                saved = json.loads((run / 'metrics.json').read_text())
                again = evaluate_food(run, self.food_root, 'cpu')
                self.assertEqual(saved['accuracy'], again['accuracy'])
                frame = food.read_data(self.food_root)
                split = json.loads((run / 'split.json').read_text())
                groups = food.image_groups(self.food_root, frame)
                self.assertFalse(set(groups[split['train_indices']]) & set(groups[split['validation_indices']]))
                self.assertFalse(set(split['train_ids']) & set(split['validation_ids']))
                run_cli('src.hw2_person.evaluate', '--data-dir', self.food_root, '--run-dir', run, '--device', 'cpu')
                output = run / 'reload.csv'
                run_cli('src.hw2_person.predict', '--data-dir', self.food_root, '--run-dir', run,
                        '--output', output, '--device', 'cpu')
                submission = pd.read_csv(output)
                pd.testing.assert_frame_equal(submission, pd.read_csv(run / 'submission.csv'))
                self.assertEqual(submission.id.tolist(), ['test_c.jpg', 'test_b.jpg', 'test_a.jpg'])
                self.assertTrue(set(submission.label).issubset(food.CLASSES))
                # 内容变化后，评估必须拒绝原来的划分。
                original = (self.food_root / 'train.csv').read_bytes()
                try:
                    frame.iloc[::-1].to_csv(self.food_root / 'train.csv', index=False)
                    with self.assertRaisesRegex(ValueError, '变化'):
                        evaluate_food(run, self.food_root, 'cpu')
                finally:
                    (self.food_root / 'train.csv').write_bytes(original)

    def test_pixel_hash_detects_identical_images(self):
        original = self.food_root / 'train_images' / '0_0.jpg'
        destination = self.food_root / 'train_images' / 'hash_duplicate.jpg'
        try:
            destination.write_bytes(original.read_bytes())
            groups = food.image_groups(self.food_root, pd.DataFrame({'id': ['0_0.jpg', destination.name]}))
            self.assertEqual(groups[0], groups[1])
        finally:
            destination.unlink(missing_ok=True)

    def test_validation_transforms_are_deterministic(self):
        frame = food.read_data(self.food_root)
        dataset = food.FoodDataset(self.food_root, frame, image_size=32)
        self.assertTrue(torch.equal(dataset[0][0], dataset[0][0]))
        self.assertEqual(tuple(dataset[0][0].shape), (3, 32, 32))

    def test_submission_rejects_wrong_ids_and_nan(self):
        with self.assertRaises(ValueError):
            wine.write_submission(self.wine_root, [100, 101, 102, 103], [1, 2, np.nan, 4], self.root / 'bad.csv')
        with self.assertRaises(ValueError):
            food.write_submission(self.food_root, ['missing.jpg'] * 3, ['burger'] * 3, self.root / 'bad.csv')


if __name__ == '__main__':
    unittest.main()
