# 个人 1：红葡萄酒质量预测

数据：`datasets/ml-hw1-person/`。在项目根目录激活 `.venv` 后运行。

## 训练、评估与预测

```bash
python -m src.hw1_person.train --output-dir outputs/hw1_person/baseline
python -m src.hw1_person.evaluate --run-dir outputs/hw1_person/baseline
python -m src.hw1_person.predict --run-dir outputs/hw1_person/baseline
```

默认模型为 `ridge`。Ridge、ExtraTrees、MLP（11→32→32→1）。--model extra_trees --trees 300；--model mlp --epochs 100；Ridge 用 --alpha 调整正则化。

重复特征绑定为组，五折 GroupKFold；每折独立拟合标准化，保存折外预测。

评估加载各折模型重算折外指标；`model.joblib` 在全量数据上重训，仅用于测试预测。

各入口支持 `--data-dir`。每次实验使用新的 `--output-dir`，程序拒绝覆盖非空实验目录。模型代码在 `models.py`，数据在 `data.py`，训练、评估、预测各有独立入口；只在本作业内复用。

## 查看结果

主指标：RMSE ↓、MAE ↓。

- `metrics.json`：本次训练的验证指标、基线、耗时；神经图像模型还记录最佳轮数。
- `evaluation.json`：独立评估入口重新加载验证模型得到的指标。
- `config.json`：参数；划分文件保存训练/验证 ID，评估会校验数据指纹。
- `history.json`：MLP 的各折及全量重训损失；Ridge 和树模型的列表为空。
- `model.joblib`：全量模型；`submission.csv`：测试集预测，按模板顺序导出。

`splits.json`、`fold_*.joblib`、`oof_predictions.csv`；后者包含真实质量、预测和残差。`evaluation.json` 包含按质量分段误差。

```bash
python -m json.tool outputs/hw1_person/baseline/metrics.json
python -m json.tool outputs/hw1_person/baseline/history.json
```

训练和预测不上传 Kaggle。
