# 个人 3：背包价格预测

数据：`datasets/ml-hw3-person/`。在项目根目录激活 `.venv` 后运行。

## 训练、评估与预测

```bash
python -m src.hw3_person.train --output-dir outputs/hw3_person/baseline
python -m src.hw3_person.evaluate --run-dir outputs/hw3_person/baseline
python -m src.hw3_person.predict --run-dir outputs/hw3_person/baseline
```

默认模型为 `ridge`。独热编码 Ridge、ExtraTrees、MLP（64/32 隐藏层）。--model extra_trees --trees 100；--model mlp --epochs 30。

约 20% 留出；若存在相同特征组合，按组划分。数值缺失用训练折中位数填充并加指示；类别缺失保留独立字符串，编码器处理未知类别。

使用原价格训练与评价。MLP 为标准化输入的 sklearn 神经网络；先比较 Ridge 和树模型。

各入口支持 `--data-dir`。每次实验使用新的 `--output-dir`，程序拒绝覆盖非空实验目录。模型代码在 `models.py`，数据在 `data.py`，训练、评估、预测各有独立入口；只在本作业内复用。

## 查看结果

主指标：RMSE ↓、MAE ↓。

- `metrics.json`：本次训练的验证指标、基线、耗时；神经图像模型还记录最佳轮数。
- `evaluation.json`：独立评估入口重新加载验证模型得到的指标。
- `config.json`：参数；划分文件保存训练/验证 ID，评估会校验数据指纹。
- `history.json`：网络训练损失及图像网络的逐轮验证指标，区分验证阶段与全量训练阶段。
- `validation_model.joblib`：保留的验证模型，不参与全量重训；`model.joblib`：全量模型；`submission.csv`：测试集预测，按模板顺序导出。

`split.json`、`validation_predictions.csv`（真实价格、预测、残差）；`metrics.json` 包含训练均值基线。

```bash
python -m json.tool outputs/hw3_person/baseline/metrics.json
python -m json.tool outputs/hw3_person/baseline/history.json
```

传统模型没有逐轮神经网络损失；相应 `history.json` 列表为空。训练和预测不上传 Kaggle。
