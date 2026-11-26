# 团队 4：扫描文档去噪修复

数据：`datasets/ml-hw4-group/`。在项目根目录激活 `.venv` 后运行。

## 训练、评估与预测

```bash
python -m src.hw4_group.train --output-dir outputs/hw4_group/baseline
python -m src.hw4_group.evaluate --run-dir outputs/hw4_group/baseline
python -m src.hw4_group.predict --run-dir outputs/hw4_group/baseline --images-only
python -m src.hw4_group.predict --run-dir outputs/hw4_group/baseline
```

训练默认只保存模型、指标和验证预览。第一条预测命令仅保存 `test_predictions/*.npy`；第二条按模板分块导出约 1130 万行 `submission.csv`，检查每个像素恰好出现一次。训练时加 `--export-submission` 可直接导出。

默认模型为 `unet`。原图直出、中值滤波、小型 U-Net（16/32/64 通道）。--model identity/median 使用传统对照；U-Net 用 --epochs、--patience、--patch-size、--patches-per-image、--batch-size、--lr 调参。

original/mixed 的同文档变体绑定为组，约 20% 文档验证。先分组再裁块；输入与目标使用相同随机裁剪。

输入输出均在 [0,1]；训练图块，验证和预测整图，自动填充再裁回原尺寸。按验证 RMSE 早停，再按最佳轮数全量重训。训练显存不足时降低 batch-size 或 patch-size；整图推理超出显存时可使用 --device cpu。

各入口支持 `--data-dir`、`--device auto/cpu/cuda` 和 `--threads`。每次实验使用新的 `--output-dir`，程序拒绝覆盖非空实验目录。模型代码在 `models.py`，数据在 `data.py`，训练、评估、预测各有独立入口；只在本作业内复用。

## 查看结果

主指标：全像素灰度 RMSE ↓、MAE ↓、PSNR ↑。

- `metrics.json`：本次训练的验证指标、基线、耗时；神经图像模型还记录最佳轮数。
- `evaluation.json`：独立评估入口重新加载验证模型得到的指标。
- `config.json`：参数；划分文件保存训练/验证 ID，评估会校验数据指纹。
- `history.json`：网络训练损失及图像网络的逐轮验证指标，区分验证阶段与全量训练阶段。
- `validation_model.joblib`：保留的验证模型，不参与全量重训；`model.joblib`：全量模型；`submission.csv`：测试集预测，按模板顺序导出。

`evaluation.json` 包含按像素加权的全局及 original/mixed 指标；`previews/` 从左到右为噪声、恢复、目标、放大 4 倍的绝对误差。

```bash
python -m json.tool outputs/hw4_group/baseline/metrics.json
python -m json.tool outputs/hw4_group/baseline/history.json
```

传统模型没有逐轮神经网络损失；相应 `history.json` 列表为空。训练和预测不上传 Kaggle。
