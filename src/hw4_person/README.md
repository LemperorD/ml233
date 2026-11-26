# 个人 4：内向／外向性格分类

数据：`datasets/ml-hw4-person/`。在项目根目录激活 `.venv` 后运行。

## 训练、评估与预测

```bash
python -m src.hw4_person.train --output-dir outputs/hw4_person/baseline
python -m src.hw4_person.evaluate --run-dir outputs/hw4_person/baseline
python -m src.hw4_person.predict --run-dir outputs/hw4_person/baseline
```

默认模型为 `logistic`。逻辑回归、ExtraTrees、MLP（32/16 隐藏层）。--model extra_trees --trees 100；--model mlp --epochs 30。

约 20% 分层留出；重复特征存在时按组留出并检查两侧类别覆盖。数值中位数填充及缺失指示、类别缺失保留为独立值。

保存并输出 Extrovert/Introvert 字符串。初版不搜索阈值，也不默认使用类别权重。

各入口支持 `--data-dir`。每次实验使用新的 `--output-dir`，程序拒绝覆盖非空实验目录。模型代码在 `models.py`，数据在 `data.py`，训练、评估、预测各有独立入口；只在本作业内复用。

## 查看结果

主指标：Accuracy ↑、Macro-F1 ↑。

- `metrics.json`：本次训练的验证指标、基线、耗时；神经图像模型还记录最佳轮数。
- `evaluation.json`：独立评估入口重新加载验证模型得到的指标。
- `config.json`：参数；划分文件保存训练/验证 ID，评估会校验数据指纹。
- `history.json`：网络训练损失及图像网络的逐轮验证指标，区分验证阶段与全量训练阶段。
- `validation_model.joblib`：保留的验证模型，不参与全量重训；`model.joblib`：全量模型；`submission.csv`：测试集预测，按模板顺序导出。

`validation_predictions.csv`、`errors.csv`、`confusion_matrix.csv`；逐类精确率、召回率在 `evaluation.json`，多数类基线在 `metrics.json`。

```bash
python -m json.tool outputs/hw4_person/baseline/metrics.json
python -m json.tool outputs/hw4_person/baseline/history.json
```

传统模型没有逐轮神经网络损失；相应 `history.json` 列表为空。训练和预测不上传 Kaggle。
