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

## 作业方法指导

八个候选题的题号映射、官方评分指标、截止时间、报告打包要求，以及任务分析、验证划分和模型路线见 [方法指导总览](assets/docs/README.md)。本地数据统计见 [数据概况](assets/docs/data_profile.json)。文档及代码统一按课程编号：个人 1 为葡萄酒，个人 2 为美食；历史数据目录名称保持不变。其余六题目前仅有方法指南。


## 个人作业代码

安装训练依赖（PyTorch 可先按上文选择 CPU/CUDA 安装源）：

```bash
python -m pip install -r requirements.txt
```

| 课程编号 | 代码及运行说明 | 实际数据目录 |
| --- | --- | --- |
| 个人 1：葡萄酒回归 | [src/hw1_person](src/hw1_person/README.md) | `datasets/ml-hw2-person/` |
| 个人 2：美食分类 | [src/hw2_person](src/hw2_person/README.md) | `datasets/ml-hw1-person/` |

```bash
python -m src.hw1_person.train --model ridge --output-dir outputs/hw1_person/ridge
python -m src.hw2_person.train --model cnn --output-dir outputs/hw2_person/cnn
python -m unittest discover -s tests -v
```

两份作业各自拥有数据读取、模型、训练、评估、预测模块，各作业的网络在内部复用，无跨作业公共模型层。训练自动保存验证指标、划分、模型及按模板对齐的 `submission.csv`，不自动上传 Kaggle。请使用不同输出目录记录不同实验。测试使用临时合成数据检查全部五种模型的训练、保存加载、独立评估、推理、重复分组隔离和提交 ID 对齐。
