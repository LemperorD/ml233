# 团队 2：作家风格识别

数据：`datasets/ml-hw2-group/`。在项目根目录激活 `.venv` 后运行。

## 训练、评估与预测

```bash
python -m src.hw2_group.train --output-dir outputs/hw2_group/baseline
python -m src.hw2_group.evaluate --run-dir outputs/hw2_group/baseline
python -m src.hw2_group.predict --run-dir outputs/hw2_group/baseline
```

默认模型为 `logistic`。TF-IDF 加逻辑回归、MultinomialNB、MLP（64 隐藏层）。--features word/char/both；--model naive_bayes；--model mlp --epochs 30。

按作者分层留出约 20%；归一化文本相同时按组留出。词表和 IDF 仅拟合训练折。

提交按 EAP,HPL,MWS 显式对齐概率列，每行和为 1。TF-IDF 保持稀疏，MLP 使用稀疏输入。

各入口支持 `--data-dir`。每次实验使用新的 `--output-dir`，程序拒绝覆盖非空实验目录。模型代码在 `models.py`，数据在 `data.py`，训练、评估、预测各有独立入口；只在本作业内复用。

## 查看结果

主指标：Log Loss ↓、Accuracy ↑。

- `metrics.json`：本次训练的验证指标、基线、耗时；神经图像模型还记录最佳轮数。
- `evaluation.json`：独立评估入口重新加载验证模型得到的指标。
- `config.json`：参数；划分文件保存训练/验证 ID，评估会校验数据指纹。
- `history.json`：网络训练损失及图像网络的逐轮验证指标，区分验证阶段与全量训练阶段。
- `validation_model.joblib`：保留的验证模型，不参与全量重训；`model.joblib`：全量模型；`submission.csv`：测试集预测，按模板顺序导出。

`validation_predictions.csv` 保留三类概率；`errors.csv`、`confusion_matrix.csv` 用于误判分析；`metrics.json` 包含类别先验基线。

```bash
python -m json.tool outputs/hw2_group/baseline/metrics.json
python -m json.tool outputs/hw2_group/baseline/history.json
```

传统模型没有逐轮神经网络损失；相应 `history.json` 列表为空。训练和预测不上传 Kaggle。
