# 个人作业 2：全球特色美食图像分类

按课程 PDF 编号。本作业读取历史数据目录 `datasets/ml-hw1-person/`。所有命令在项目根目录、激活 `.venv` 后执行；依赖见根目录 `requirements.txt`。实现不依赖 torchvision、不使用预训练权重、不下载外部数据。

## 文件职责

- `data.py`：CSV/图片校验、固定十类映射、RGB 像素哈希分组、PIL 增强、颜色直方图和提交对齐。
- `models.py`：`FoodCNN` 和直方图逻辑回归；网络只在本作业内复用。
- `engine.py`：本作业内共用的种子、批量读取、训练、早停与概率预测。
- `train.py`：分层分组留出、训练与验证、按最佳轮数全量重训、生成提交。
- `evaluate.py`：加载独立验证模型，输出 Accuracy、Macro-F1、逐类报告与混淆矩阵。
- `predict.py`：加载全量模型，为测试集生成英文标签提交。

## 训练与比较

```bash
python -m src.hw2_person.train --model histogram --output-dir outputs/hw2_person/histogram
python -m src.hw2_person.train --model cnn --epochs 20 --batch-size 32 --output-dir outputs/hw2_person/cnn
```

默认 `--device auto`，可指定 `--device cpu` 或 `--device cuda`。默认 `--threads 2` 控制 CPU 线程数，`--image-size 128`、`--lr 0.001`、`--patience 5`。快速检查可使用 `--epochs 1 --image-size 64`；一轮结果只用于验证流程。

默认种子 42，无重复图片时分层随机留出；有重复时以 StratifiedGroupKFold 的第一折作验证。默认 `--folds 5`，约留出 20%。相同解码 RGB 像素的图片绑定为一组；近重复图片尚未通过感知哈希识别，后续实验应补查。两种模型相同参数下共享划分，保存 ID 与图片/CSV 指纹；训练和验证都需覆盖十类，否则明确报错。

CNN 为四组 `Conv2d → BatchNorm → ReLU → MaxPool`，通道为 16/32/64/128，随后全局平均池化、Dropout 和十分类线性层，使用交叉熵与 AdamW。训练采用温和随机裁剪、翻转、亮度/饱和度扰动；验证和推理保留整图比例并填充至方形，固定映射到 `[-1,1]`。直方图基线为三个颜色通道各 16 桶，训练折标准化后做逻辑回归。

CNN 按验证 Accuracy 保存最佳轮次并早停，随后固定轮数从头全量重训。`validation_model.joblib` 永久保留供评估；`model.joblib` 用于测试推理。直方图同样分别保存训练折模型和全量模型。实验目录必须为空，避免覆盖或混用产物。

## 独立评估与预测

```bash
python -m src.hw2_person.evaluate --run-dir outputs/hw2_person/cnn
python -m src.hw2_person.predict --run-dir outputs/hw2_person/cnn --output outputs/hw2_person/cnn/submission.csv
```

两个入口支持 `--device`、`--threads`，所有入口支持 `--data-dir /path/to/food`。评估重算验证集结果，保存 `evaluation.json`、`confusion_matrix.csv`、`validation_predictions.csv`、`errors.csv`；错误清单保留原图 ID，便于查看背景干扰、裁剪和相似食物。训练额外保存配置、类别顺序、划分、训练历史、指标及提交。`history.json` 中区分验证阶段与全量训练阶段。

提交按模板顺序写 `id,label`，使用本地真实标签 `burger`，检查每张测试图恰好预测一次。只加载自己生成或可信来源的 `.joblib` 模型。运行不会上传 Kaggle；本地验证分数不等于排行榜分数。

## 本地验证记录

2026-11-26，使用本地完整数据、种子 42、约 20% 分层留出，直方图基线 Accuracy 为 0.273224、Macro-F1 为 0.261148。一轮 CNN（64×64、CPU）Accuracy 为 0.142077，仅用于训练和加载流程检查，尚未调优。两种模型均已生成 608 行提交文件；尚未取得 Kaggle 分数。
