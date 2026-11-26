# 团队 3：人脸关键点定位

数据：`datasets/ml-hw3-group/`。在项目根目录激活 `.venv` 后运行。

## 训练、评估与预测

```bash
python -m src.hw3_group.train --output-dir outputs/hw3_group/baseline
python -m src.hw3_group.evaluate --run-dir outputs/hw3_group/baseline
python -m src.hw3_group.predict --run-dir outputs/hw3_group/baseline
```

默认模型为 `cnn`。坐标均值、小型 CNN（16/32/64 卷积，128 隐藏层，30 维输出）。--model mean 使用均值对照；CNN 用 --epochs、--patience、--batch-size、--lr 调参。

完全相同图片绑定为组；无重复时平衡完整与稀疏标注，留出约 20%。缺失坐标通过掩码忽略，不替换为真实零标签。

坐标按 (x-48)/48 归一化，预测恢复到像素单位。仅做亮度增强；按验证 RMSE 早停，再按最佳轮数全量重训。提交通过 IdLookupTable.csv 查表，不能直接展平。

各入口支持 `--data-dir`、`--device auto/cpu/cuda` 和 `--threads`。每次实验使用新的 `--output-dir`，程序拒绝覆盖非空实验目录。模型代码在 `models.py`，数据在 `data.py`，训练、评估、预测各有独立入口；只在本作业内复用。

## 查看结果

主指标：有效坐标像素 RMSE ↓。

- `metrics.json`：本次训练的验证指标、基线、耗时；神经图像模型还记录最佳轮数。
- `evaluation.json`：独立评估入口重新加载验证模型得到的指标。
- `config.json`：参数；划分文件保存训练/验证 ID，评估会校验数据指纹。
- `history.json`：网络训练损失及图像网络的逐轮验证指标，区分验证阶段与全量训练阶段。
- `validation_model.joblib`：保留的验证模型，不参与全量重训；`model.joblib`：全量模型；`submission.csv`：测试集预测，按模板顺序导出。

`validation_predictions.csv` 为 30 维像素坐标；`evaluation.json` 包含逐坐标误差、有效标注数和完整/稀疏分组误差；`previews/` 红点是真值，绿点是预测。

```bash
python -m json.tool outputs/hw3_group/baseline/metrics.json
python -m json.tool outputs/hw3_group/baseline/history.json
```

传统模型没有逐轮神经网络损失；相应 `history.json` 列表为空。训练和预测不上传 Kaggle。
