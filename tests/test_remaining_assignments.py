"""其余六项作业的独立 CLI、模型恢复、缺失掩码和流式导出检查。"""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import torch

PROJECT = Path(__file__).resolve().parents[1]


def cli(name, command, root, run, *extra):
    flag = '--output-dir' if command == 'train' else '--run-dir'
    result = subprocess.run([sys.executable, '-m', f'src.{name}.{command}', '--data-dir', str(root),
                             flag, str(run), *extra], cwd=PROJECT, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)


class RemainingAssignmentsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rng = np.random.default_rng(7)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def table_fixture(self, name):
        data = importlib.import_module(f'src.{name}.data')
        root = self.root / name
        root.mkdir()
        n = 90
        if name == 'hw2_group':
            rows = [{'id': str(i), 'text': f'{data.CLASSES[i % 3]} characteristic prose style number {i}',
                     'author': data.CLASSES[i % 3]} for i in range(n)]
            frame = pd.DataFrame(rows)
        else:
            frame = pd.DataFrame({column: self.rng.normal(size=n) for column in data.FEATURES if column in data.NUMERIC})
            for column in data.CATEGORICAL:
                frame[column] = np.where(np.arange(n) % 2, 'Yes', 'No')
            if name == 'hw1_group':
                frame[data.ID] = [f'2011-{1 + i // 45:02d}-{1 + i % 15:02d} {i // 15 % 3:02d}:00:00' for i in range(n)]
                frame[data.TARGET] = np.arange(n) + 3
                frame['casual'] = frame[data.TARGET]  # 必须被 prepare 排除。
                frame['registered'] = 0
            else:
                frame.insert(0, data.ID, np.arange(n))
                frame[data.TARGET] = (np.where(np.arange(n) % 2, 'Introvert', 'Extrovert')
                                      if name == 'hw4_person' else self.rng.uniform(10, 100, n))
                frame.loc[0, data.NUMERIC[0]] = np.nan
                frame.loc[1, data.CATEGORICAL[0]] = np.nan
        frame.to_csv(root / 'train.csv', index=False)
        test = frame.drop(columns=[data.TARGET]).iloc[:4].copy()
        test.to_csv(root / 'test.csv', index=False)
        template = pd.DataFrame({data.ID: test[data.ID].iloc[::-1].to_numpy()})
        if name == 'hw2_group':
            for label in data.CLASSES:
                template[label] = 1 / 3
        else:
            template[data.TARGET] = data.CLASSES[0] if data.CLASSES else 0.0
        template.to_csv(root / data.TEMPLATE, index=False)
        return root, data, frame

    def test_four_table_tasks_cli_and_all_models(self):
        tasks = {'hw3_person': ['ridge', 'extra_trees', 'mlp'], 'hw4_person': ['logistic', 'extra_trees', 'mlp'],
                 'hw1_group': ['gradient_boosting', 'random_forest', 'mlp'], 'hw2_group': ['logistic', 'naive_bayes', 'mlp']}
        for name, names in tasks.items():
            with self.subTest(name=name):
                root, data, frame = self.table_fixture(name)
                run = self.root / f'{name}_run'
                cli(name, 'train', root, run, '--trees', '5', '--threads', '1')
                cli(name, 'evaluate', root, run)
                before = (run / 'submission.csv').read_bytes()
                cli(name, 'predict', root, run)
                self.assertEqual(before, (run / 'submission.csv').read_bytes())
                submission = pd.read_csv(run / 'submission.csv')
                self.assertEqual(submission[data.ID].tolist(), pd.read_csv(root / data.TEMPLATE)[data.ID].tolist())
                split = json.loads((run / 'split.json').read_text())
                self.assertFalse(set(split['train_ids']) & set(split['validation_ids']))
                if name == 'hw1_group':
                    self.assertTrue((pd.to_datetime(frame.iloc[split['train_indices']][data.ID]).dt.day <= 11).all())
                    self.assertTrue((pd.to_datetime(frame.iloc[split['validation_indices']][data.ID]).dt.day >= 12).all())
                    self.assertNotIn('casual', data.prepare(frame))
                models = importlib.import_module(f'src.{name}.models')
                for model_name in names:
                    with self.subTest(model=model_name), threadpool_limits(limits=1), warnings.catch_warnings():
                        warnings.simplefilter('ignore', ConvergenceWarning)
                        model = models.build_model(model_name, seed=42, epochs=2, trees=5)
                        model.fit(data.prepare(frame), models.training_target(frame[data.TARGET]))
                        file = self.root / f'{name}_{model_name}.joblib'
                        joblib.dump(model, file)
                        predictions = models.predict_values(joblib.load(file), data.prepare(frame.iloc[:4]))
                        self.assertEqual(len(predictions), 4)
                        if name == 'hw2_group':
                            np.testing.assert_allclose(predictions.sum(axis=1), 1)
                        elif not data.CLASSES:
                            self.assertTrue(np.isfinite(predictions).all())

    def face_fixture(self):
        root = self.root / 'face'
        root.mkdir()
        points = [f'point{i}' for i in range(15)]
        mapping = pd.DataFrame({'keypoint': points, 'x_column': [f'{x}_x' for x in points], 'y_column': [f'{x}_y' for x in points]})
        mapping.to_csv(root / 'keypoints.csv', index=False)
        columns = [v for x in points for v in [f'{x}_x', f'{x}_y']]
        for folder in ['train_images/train_images', 'test_images/test_images']:
            (root / folder).mkdir(parents=True)
        train = pd.DataFrame(self.rng.uniform(10, 85, (12, 30)), columns=columns)
        train.loc[0, columns[0]] = np.nan
        train.insert(0, 'image_path', [f'train_images/{i}.png' for i in range(12)])
        train.insert(0, 'ImageId', [f'face_{i}' for i in range(12)])
        train.to_csv(root / 'train.csv', index=False)
        for i in range(12):
            Image.fromarray(self.rng.integers(0, 256, (96, 96), dtype=np.uint8)).save(root / 'train_images/train_images' / f'{i}.png')
        test = pd.DataFrame({'ImageId': ['test_a', 'test_b'], 'image_path': ['test_images/a.png', 'test_images/b.png']})
        test.to_csv(root / 'test.csv', index=False)
        for name in ['a', 'b']:
            Image.new('L', (96, 96), 100).save(root / 'test_images/test_images' / f'{name}.png')
        lookup = pd.DataFrame({'RowId': [1, 2, 3], 'ImageId': ['test_b', 'test_a', 'test_b'], 'FeatureName': [columns[3], columns[0], columns[29]]})
        lookup.to_csv(root / 'IdLookupTable.csv', index=False)
        pd.DataFrame({'RowId': [3, 1, 2], 'Location': 0.0}).to_csv(root / 'sample_submission.csv', index=False)
        return root

    def test_face_cnn_mean_mask_and_lookup(self):
        from src.hw3_group.data import read_data, columns, write_submission
        from src.hw3_group.engine import train_model, predict_values
        from src.hw3_group.models import masked_mse
        root = self.face_fixture()
        run = self.root / 'face_run'
        cli('hw3_group', 'train', root, run, '--epochs', '1', '--threads', '1', '--device', 'cpu')
        cli('hw3_group', 'evaluate', root, run, '--device', 'cpu')
        cli('hw3_group', 'predict', root, run, '--device', 'cpu')
        self.assertEqual(pd.read_csv(run / 'submission.csv').RowId.tolist(), [3, 1, 2])
        loss = masked_mse(torch.tensor([[1.0, 5.0]], requires_grad=True), torch.tensor([[2.0, float('nan')]]), torch.tensor([[True, False]]))
        self.assertEqual(float(loss.detach()), 1)
        loss.backward()
        train = read_data(root)
        mean, _, _ = train_model(root, train, {'model': 'mean', 'seed': 42})
        prediction = predict_values(mean, root, train)
        np.testing.assert_allclose(prediction[0], train[columns(root)].mean().to_numpy())
        test = read_data(root, False)
        known = np.arange(60).reshape(2, 30)
        write_submission(root, test, known, run / 'known.csv')
        self.assertEqual(pd.read_csv(run / 'known.csv').Location.tolist(), [59, 33, 0])

    def document_fixture(self):
        root = self.root / 'document'
        root.mkdir()
        for folder in ['train_noisy', 'train_clean', 'test_noisy']:
            (root / folder).mkdir()
        rows = []
        for i in range(6):
            clean = self.rng.integers(100, 256, (17 + i % 2 * 4, 19), dtype=np.uint8)
            for variant in ['original', 'mixed']:
                name = f'doc_{i}_{variant}'
                rows.append({'image_id': name, 'height': clean.shape[0], 'width': clean.shape[1], 'variant': variant})
                Image.fromarray(clean).save(root / 'train_clean' / f'{name}.png')
                Image.fromarray(np.maximum(clean.astype(int) - 20, 0).astype(np.uint8)).save(root / 'train_noisy' / f'{name}.png')
        pd.DataFrame(rows).to_csv(root / 'train_manifest.csv', index=False)
        test = pd.DataFrame([{'image_id': 'doc_test_original', 'height': 17, 'width': 19, 'variant': 'original'}])
        test.to_csv(root / 'test_manifest.csv', index=False)
        Image.new('L', (19, 17), 150).save(root / 'test_noisy/doc_test_original.png')
        ids = [f'doc_test_original_{i}_{j}' for i in range(1, 18) for j in range(1, 20)]
        pd.DataFrame({'id': ids[::-1], 'value': 0.0}).to_csv(root / 'sample_submission.csv', index=False)
        return root

    def test_document_unet_pairs_groups_and_streaming_export(self):
        from src.hw4_group.data import read_data, document_groups, write_submission, PatchDataset
        from src.hw4_group.engine import restore_image
        root = self.document_fixture()
        run = self.root / 'document_run'
        cli('hw4_group', 'train', root, run, '--epochs', '1', '--patch-size', '16', '--patches-per-image', '1',
            '--batch-size', '2', '--threads', '1', '--device', 'cpu', '--export-submission')
        cli('hw4_group', 'evaluate', root, run, '--device', 'cpu')
        cli('hw4_group', 'predict', root, run, '--device', 'cpu', '--chunksize', '37')
        frame = read_data(root)
        split = json.loads((run / 'split.json').read_text())
        groups = document_groups(frame)
        self.assertFalse(set(groups[split['train_indices']]) & set(groups[split['validation_indices']]))
        inputs, targets = PatchDataset(root, frame, patch_size=16, patches_per_image=1)[0]
        np.testing.assert_allclose((targets - inputs).numpy(), np.full((1, 16, 16), 20 / 255), atol=1e-6)
        test = read_data(root, False)
        known = np.linspace(0, 1, 17 * 19, dtype=np.float32).reshape(17, 19)
        path = run / 'known.csv'
        count = write_submission(root, test, {'doc_test_original': known}, path, chunksize=37)
        self.assertEqual(count, 17 * 19)
        np.testing.assert_allclose(pd.read_csv(path).value, known.ravel()[::-1], atol=1e-7)
        for kind in ['identity', 'median']:
            restored = restore_image({'kind': kind}, known)
            self.assertEqual(restored.shape, known.shape)
        template = pd.read_csv(root / 'sample_submission.csv')
        template.loc[0, 'id'] = template.loc[1, 'id']
        template.to_csv(root / 'sample_submission.csv', index=False)
        with self.assertRaisesRegex(ValueError, '重复像素'):
            write_submission(root, test, {'doc_test_original': known}, run / 'bad.csv', chunksize=37)
        self.assertFalse((run / 'bad.csv').exists())
        self.assertFalse((run / 'bad.csv.part').exists())

    def test_numbering_matches_data_directories(self):
        for number in range(1, 5):
            for kind in ['person', 'group']:
                data = importlib.import_module(f'src.hw{number}_{kind}.data')
                self.assertEqual(data.DEFAULT_DATA.name, f'ml-hw{number}-{kind}')


if __name__ == '__main__':
    unittest.main()
