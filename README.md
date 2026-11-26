# ml233

机器学习课程作业：个人作业四选一，团队作业四选一（2–3 人一组）。要求依据 [2026 年作业说明 PDF](assets/docs/2026研究生机器学习大作业说明.pdf)。

## 环境准备

在项目根目录创建并激活独立环境，后续安装与运行均使用该环境的 `python`：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn pillow kagglehub
```

- 注册 Kaggle 账户并配置 API Token，参考 [KaggleHub 认证说明](https://github.com/Kaggle/kagglehub#authenticate)。
- 如需交互式配置认证，执行 `python -c "import kagglehub; kagglehub.login()"`。

### 安装 PyTorch 与检测 CUDA

当前项目的 Python 3.10 环境使用 CUDA 12.8 版 PyTorch，从官方源安装：

```bash
source .venv/bin/activate
python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip check
```

其他机器应根据显卡与驱动选择安装版本，参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)。`nvidia-smi` 显示的是驱动支持的 CUDA 版本，`torch.version.cuda` 显示的是 PyTorch 构建使用的 CUDA 版本，两者不必完全一致。

快速检测 CUDA 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

输出 `True` 表示当前进程可以使用 CUDA。进一步查看版本、显卡，并实际执行一次 GPU 矩阵运算：

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
available = torch.cuda.is_available()
print("CUDA available:", available)
print("GPU count:", torch.cuda.device_count())
if available:
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(256, 256, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    print("GPU matrix multiplication:", y.shape, y.device)
PY
```

若输出 `False`，先运行 `nvidia-smi` 检查驱动和当前运行环境是否能访问显卡，再确认 `torch.version.cuda` 不是 `None`（`None` 表示安装了 CPU 版）。容器或沙箱还需要获得 GPU 访问权限。

## 下载数据集

```bash
python src/install_datasets.py
```

脚本按顺序下载八个数据集，保存到项目根目录下的 `datasets/`。仅下载部分作业时，注释其他作业对应的代码块。脚本默认使用已经配置的 Kaggle 凭据，不会每次提示登录。

若下载出现 `storage.googleapis.com` 连接超时，请检查终端的网络或代理设置，然后重新运行。不要将 API Token 提交到仓库。

## 作业与代码

任务分析和课程要求见[方法指南](assets/docs/README.md)。八项代码相互独立，每项均包含 `data.py`、`models.py`、`train.py`、`evaluate.py`、`predict.py`；网络与计算过程在各自作业内复用。

```bash
python -m pip install -r requirements.txt
```

| 作业 | 数据目录 | 模型与详细说明 | 主指标 |
| --- | --- | --- | --- |
| 个人 1：葡萄酒回归 | `datasets/ml-hw1-person/` | [Ridge / ExtraTrees / MLP](src/hw1_person/README.md) | RMSE ↓ |
| 个人 2：美食分类 | `datasets/ml-hw2-person/` | [直方图 / CNN](src/hw2_person/README.md) | Accuracy ↑ |
| 个人 3：背包价格 | `datasets/ml-hw3-person/` | [Ridge / ExtraTrees / MLP](src/hw3_person/README.md) | RMSE ↓ |
| 个人 4：性格分类 | `datasets/ml-hw4-person/` | [逻辑回归 / ExtraTrees / MLP](src/hw4_person/README.md) | Accuracy ↑ |
| 团队 1：单车需求 | `datasets/ml-hw1-group/` | [梯度提升 / 随机森林 / MLP](src/hw1_group/README.md) | RMSLE ↓ |
| 团队 2：作者识别 | `datasets/ml-hw2-group/` | [词、字符 TF-IDF + 分类器](src/hw2_group/README.md) | Log Loss ↓ |
| 团队 3：人脸关键点 | `datasets/ml-hw3-group/` | [坐标均值 / 掩码 CNN](src/hw3_group/README.md) | 像素 RMSE ↓ |
| 团队 4：文档去噪 | `datasets/ml-hw4-group/` | [原图 / 中值滤波 / U-Net](src/hw4_group/README.md) | [0,1] 灰度 RMSE ↓ |

## 训练八项作业

在项目根目录运行，每条命令是独立实验：

```bash
python -m src.hw1_person.train
python -m src.hw2_person.train
python -m src.hw3_person.train
python -m src.hw4_person.train
python -m src.hw1_group.train
python -m src.hw2_group.train
python -m src.hw3_group.train
python -m src.hw4_group.train
```

默认输出为 `outputs/<作业>/baseline/`。使用 `--output-dir outputs/<作业>/<新实验名>` 保存新的实验，非空目录不覆盖。`--model` 选择模型，具体选择及参数见表中各作业说明。图像网络默认最多 20 轮并早停，验证选出轮数后从头全量重训；表格和文本 MLP 默认 30 轮，`ConvergenceWarning` 表示达到设置轮数时尚未收敛，后续应检查损失并增加轮数。

图像任务支持 `--device auto/cpu/cuda`，默认自动选择；当前无 GPU 时可先检查一轮：

```bash
python -m src.hw2_person.train --epochs 1 --image-size 64 --device cpu --output-dir outputs/hw2_person/check
python -m src.hw3_group.train --epochs 1 --device cpu --output-dir outputs/hw3_group/check
python -m src.hw4_group.train --epochs 1 --patch-size 64 --patches-per-image 1 --device cpu --output-dir outputs/hw4_group/check
```

文档去噪默认不导出约 1130 万行提交；可先查看验证预览，再单独预测和导出。其他作业训练结束会生成测试提交。单车验证使用每月 12–15 日；葡萄酒按重复特征分组五折；其余按各作业的分层或分组策略留出验证，模型预处理仅在训练部分拟合。

## 独立评估与生成提交

将模块名和实验目录一起替换即可评估任一作业，例如：

```bash
python -m src.hw1_person.evaluate --run-dir outputs/hw1_person/baseline
python -m src.hw1_person.predict --run-dir outputs/hw1_person/baseline
python -m src.hw3_group.evaluate --run-dir outputs/hw3_group/baseline --device cpu
python -m src.hw4_group.predict --run-dir outputs/hw4_group/baseline --images-only
python -m src.hw4_group.predict --run-dir outputs/hw4_group/baseline
```

评估重载保留的验证模型；葡萄酒重载各折模型。全量重训模型仅用于测试推理。预测按模板对齐 ID、检查输出格式并生成 `submission.csv`；关键点按查找表提取坐标，文档去噪按模板流式导出像素。代码不会自动上传 Kaggle。

## 查看训练与评估结果

| 文件 | 用途 |
| --- | --- |
| `config.json` | 查看本次模型与训练参数 |
| `split.json` / 葡萄酒的 `splits.json` | 查看训练、验证 ID，确认实验划分 |
| `history.json` | 查看逐轮训练损失和图像模型验证指标；传统模型无逐轮损失 |
| `metrics.json` | 查看验证主指标、基线、耗时与最佳轮数（适用时） |
| `evaluation.json` | 查看独立加载模型后重算的验证指标 |
| `validation_predictions.csv` / 葡萄酒的 `oof_predictions.csv` | 对照真实值与预测，检查分类错误或回归残差 |
| `confusion_matrix.csv`、`errors.csv` | 分类任务查看混淆类别与错分样本 |
| `previews/` | 关键点：红色真值、绿色预测；去噪：噪声、恢复、目标、误差 |
| `submission.csv` | 最终测试预测；文档浮点恢复图另存于 `test_predictions/*.npy` |

```bash
python -m json.tool outputs/hw2_person/baseline/metrics.json
python -m json.tool outputs/hw2_person/baseline/history.json
python -m json.tool outputs/hw2_person/baseline/evaluation.json
```

葡萄酒 MLP 的各折损失在 `history.json` 的 `folds` 中，也保存在模型末层的 `loss_history_`。分类先看官方主指标，再看少数类召回与混淆矩阵；回归同时查看残差及不同目标区间的误差。只在相同数据和划分下比较实验。当前初版及短程检查结果见[本地验证记录](assets/docs/initial-results.md)，没有使用排行榜分数选模。

## 运行检查

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

检查使用临时合成数据，覆盖八项作业入口、模型保存加载、缺失处理、分组隔离、时序划分、概率列顺序、关键点查表和分块像素导出。训练产物和数据不纳入 Git。
