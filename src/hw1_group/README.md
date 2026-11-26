# 团队 1：共享单车租赁需求预测

数据：`datasets/ml-hw1-group/`。在项目根目录激活 `.venv` 后运行。

## 训练、评估与预测

```bash
python -m src.hw1_group.train --output-dir outputs/hw1_group/baseline
python -m src.hw1_group.evaluate --run-dir outputs/hw1_group/baseline
python -m src.hw1_group.predict --run-dir outputs/hw1_group/baseline
```

默认模型为 `gradient_boosting`。梯度提升、随机森林、MLP（64/32 隐藏层）。--model random_forest --trees 100；--model mlp --epochs 30；--trees 也控制梯度提升迭代数。

每月 1–11 日训练、12–15 日验证，按日期隔离。特征包含小时、星期、年月、周期编码与天气；排除 casual、registered 及目标滞后。

训练 log1p(count)，推理 expm1 后截到非负，按原计数计算 RMSLE。初版不使用随机逐行验证。

各入口支持 `--data-dir`。每次实验使用新的 `--output-dir`，程序拒绝覆盖非空实验目录。模型代码在 `models.py`，数据在 `data.py`，训练、评估、预测各有独立入口；只在本作业内复用。

## 查看结果

主指标：RMSLE ↓、计数 RMSE ↓。

- `metrics.json`：本次训练的验证指标、基线、耗时；神经图像模型还记录最佳轮数。
- `evaluation.json`：独立评估入口重新加载验证模型得到的指标。
- `config.json`：参数；划分文件保存训练/验证 ID，评估会校验数据指纹。
- `history.json`：网络训练损失及图像网络的逐轮验证指标，区分验证阶段与全量训练阶段。
- `validation_model.joblib`：保留的验证模型，不参与全量重训；`model.joblib`：全量模型；`submission.csv`：测试集预测，按模板顺序导出。

`split.json` 保存日期清单；`validation_predictions.csv` 保存原计数、非负预测及残差。

```bash
python -m json.tool outputs/hw1_group/baseline/metrics.json
python -m json.tool outputs/hw1_group/baseline/history.json
```

传统模型没有逐轮神经网络损失；相应 `history.json` 列表为空。训练和预测不上传 Kaggle。
