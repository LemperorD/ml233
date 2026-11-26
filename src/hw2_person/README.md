# 个人 2：全球特色美食图像分类

数据：`datasets/ml-hw2-person/`。在项目根目录激活 `.venv` 后运行。

## 训练、评估与预测

```bash
python -m src.hw2_person.train --output-dir outputs/hw2_person/baseline
python -m src.hw2_person.evaluate --run-dir outputs/hw2_person/baseline
python -m src.hw2_person.predict --run-dir outputs/hw2_person/baseline
```

默认模型为 `cnn`。颜色直方图加逻辑回归、四层小型 CNN（16/32/64/128 通道）。--model histogram 使用直方图；CNN 用 --epochs、--patience、--lr、--batch-size、--image-size 调参。

固定十类映射；像素相同图片绑定为组。有重复时分层分组验证，否则分层留出约 20%。训练增强不应用于验证。

CNN 按验证 Accuracy 早停并保存最佳模型，再按最佳轮数从头全量重训。无需预训练或 torchvision。

各入口支持 `--data-dir`、`--device auto/cpu/cuda` 和 `--threads`。每次实验使用新的 `--output-dir`，程序拒绝覆盖非空实验目录。模型代码在 `models.py`，数据在 `data.py`，训练、评估、预测各有独立入口；只在本作业内复用。

## 查看结果

主指标：Accuracy ↑、Macro-F1 ↑。

- `metrics.json`：本次训练的验证指标、基线、耗时；神经图像模型还记录最佳轮数。
- `evaluation.json`：独立评估入口重新加载验证模型得到的指标。
- `config.json`：参数；划分文件保存训练/验证 ID，评估会校验数据指纹。
- `history.json`：网络训练损失及图像网络的逐轮验证指标，区分验证阶段与全量训练阶段。
- `validation_model.joblib`：保留的验证模型，不参与全量重训；`model.joblib`：全量模型；`submission.csv`：测试集预测，按模板顺序导出。

`class_to_idx.json`、`split.json`、`validation_predictions.csv`、`errors.csv`、`confusion_matrix.csv`。错误清单中的 id 对应原图。

```bash
python -m json.tool outputs/hw2_person/baseline/metrics.json
python -m json.tool outputs/hw2_person/baseline/history.json
```

传统模型没有逐轮神经网络损失；相应 `history.json` 列表为空。训练和预测不上传 Kaggle。
