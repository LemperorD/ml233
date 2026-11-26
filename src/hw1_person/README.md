# 个人作业 1：红葡萄酒质量预测

按课程 PDF 编号。本作业读取历史数据目录 `datasets/ml-hw2-person/`，不是美食数据。所有命令在项目根目录、激活 `.venv` 后执行；依赖见根目录 `requirements.txt`，PyTorch 安装方式见根 README。

## 文件职责

- `data.py`：十一项特征读取与校验、重复特征分组、CSV 指纹和提交 ID 对齐。
- `models.py`：Ridge、ExtraTrees、小型 `WineMLP` 及其 sklearn 包装；网络只在本作业内复用。
- `train.py`：五折 GroupKFold、训练折预处理、保存各折模型、全量重训与提交生成。
- `evaluate.py`：从各折模型重算折外 RMSE、MAE、按质量分段误差及残差。
- `predict.py`：加载全量模型，生成 `id,quality` 提交，不四舍五入或截断预测。

## 训练与比较

```bash
python -m src.hw1_person.train --model ridge --alpha 10 --output-dir outputs/hw1_person/ridge_a10
python -m src.hw1_person.train --model extra_trees --trees 300 --output-dir outputs/hw1_person/trees
python -m src.hw1_person.train --model mlp --epochs 100 --output-dir outputs/hw1_person/mlp
```

默认种子 42；分组五折由特征组决定，三种模型采用相同划分。Ridge/MLP 的标准化只在各训练折拟合；MLP 同样仅使用训练折的目标均值，优化 MSE。默认网络为 `11 → 32 → 32 → 1`，隐藏层 ReLU。小数据优先比较 Ridge 和树模型，MLP 作为神经网络对照。`--alpha` 仅用于 Ridge，`--trees` 仅用于 ExtraTrees，`--epochs` 仅用于 MLP。修改超参数时使用新的实验目录，程序拒绝覆盖非空目录。

每次训练自动保存：`config.json`、`splits.json`（包含行索引和 ID）、`fold_*.joblib`、`model.joblib`、`metrics.json`、`evaluation.json`、`oof_predictions.csv`、`submission.csv`。`metrics.json` 同时记录训练折均值基线、重复行与冲突标签组数。模型选择依据折外 RMSE；当前每个命令只评估指定配置，需自行比较不同实验，程序不会按训练分数选模。

## 独立评估与预测

```bash
python -m src.hw1_person.evaluate --run-dir outputs/hw1_person/ridge_a10
python -m src.hw1_person.predict --run-dir outputs/hw1_person/ridge_a10 --output outputs/hw1_person/ridge_a10/submission.csv
```

各入口均可通过 `--data-dir /path/to/wine` 指定数据目录。评估会校验训练 CSV 内容与顺序，拒绝将历史划分套到变化后的数据。提交依据模板 ID 顺序导出，检查列名、唯一 ID、有限预测值及测试 ID 集合。

只加载自己生成或可信来源的 `.joblib` 模型。运行不会上传 Kaggle；排行榜分数需实际提交后另行记录。

## 本地验证记录

2026-11-26，使用本地完整数据、默认 Ridge（alpha=10）、五折重复特征分组验证，折外 RMSE 为 0.629397，MAE 为 0.491676；均值基线 RMSE 为 0.804657。已生成 480 行提交文件。这是本地验证，尚未取得 Kaggle 分数。
