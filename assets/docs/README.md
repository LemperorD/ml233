# 八个机器学习作业的方法总览

先完成固定验证集和正确的提交文件，再逐项改进模型。八份作业的数据形式不同，不能共用一套随机划分或一个评价指标。下面的指南依据本地数据与提交模板编写，数据快照日期为 2026 年 11 月 26 日；统计明细见 [数据概况](data_profile.json)。

## 作业索引

| 作业 | 数据目录 | 任务 | 训练与测试规模 | 推荐起点 |
| --- | --- | --- | --- | --- |
| [个人一](hw1-person.md) | `ml-hw1-person` | 十类食物图像分类 | 912 与 608 张 | 冻结图像特征加线性分类器 |
| [个人二](hw2-person.md) | `ml-hw2-person` | 葡萄酒质量预测 | 1119 与 480 行 | 标准化加 Ridge 回归 |
| [个人三](hw3-person.md) | `ml-hw3-person` | 背包价格回归 | 240000 与 60000 行 | 类别特征加 CatBoost |
| [个人四](hw4-person.md) | `ml-hw4-person` | 内外向二分类 | 14819 与 3705 行 | 缺失值处理加逻辑回归 |
| [小组一](hw1-group.md) | `ml-hw1-group` | 每小时单车需求回归 | 8600 与 2286 行 | 时间特征加梯度提升树 |
| [小组二](hw2-group.md) | `ml-hw2-group` | 英文文本作者三分类 | 14684 与 4895 行 | 词和字符 TF-IDF 加逻辑回归 |
| [小组三](hw3-group.md) | `ml-hw3-group` | 人脸十五个关键点定位 | 5639 与 1410 张 | 带缺失掩码的 CNN 坐标回归 |
| [小组四](hw4-group.md) | `ml-hw4-group` | 文档灰度图像去噪 | 232 对训练图与 56 张测试图 | 传统去噪对照加小型 U-Net |

个人二、三、四以及小组一、二可以先使用 CPU。图像分类、关键点和 U-Net 的迭代更适合 GPU；没有 GPU 时先做冻结特征、坐标均值和传统滤波基线。实际耗时取决于硬件，本文不提供未实测的训练时间或分数。

## 开始训练前确认规则

八个竞赛的 Overview、Evaluation 和 Rules 页面本次未能读取。各指南中列出的指标都是本地评估建议，不能直接视为官方评分规则。开始调参前逐个确认并记录：官方指标、课程指定算法、是否允许预训练权重和外部数据、提交限制、报告要求与截止时间。竞赛入口在各指南末尾。

尤其需要确认：葡萄酒质量按回归还是分类评分；单车按 RMSE 还是 RMSLE 评分；图像分类与内外向任务是否按 Accuracy 评分；关键点与文档像素是否按 RMSE 及何种坐标或像素尺度评分。提交模板能够确定列名，不能单独确定这些规则。

## 建立独立环境

当前系统 `python3` 导入 pandas 会出现 `numpy.dtype size changed`，说明现有 pandas 和 NumPy 的二进制构建不兼容。建议在项目内新建环境，再在环境内安装依赖；不要混用系统 apt 的 pandas 与用户目录 pip 的 NumPy。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn pillow kagglehub
python -c "import numpy, pandas, sklearn; print(numpy.__version__, pandas.__version__, sklearn.__version__)"
```

表格增强模型按需安装 `catboost`。图像任务按 [PyTorch 安装向导](https://pytorch.org/get-started/locally/) 选择适合当前驱动的 PyTorch 和 torchvision；传统去噪按需安装 `scikit-image`。安装成功并跑通基线后，将实际环境导出到自己的实验目录：`python -m pip freeze > requirements.txt`。这些命令是执行指导，本次没有安装训练依赖或启动训练。

## 统一实验流程

1. 检查 ID 唯一性、缺失值、目标分布、文件存在性和训练测试字段差异。读取数据时从项目根目录开始，保留原始数据不变。
2. 固定种子，例如 `42`，保存训练与验证 ID。先划分，再拟合填充器、标准化、词表与编码器。重复样本、同一原始文档的变体必须留在同一折。
3. 做一个简单基线并生成本地提交文件。检查列名、行数、ID 顺序、取值范围和缺失值；不自动上传。
4. 每次只改变一项主要设置。记录配置、验证集分数、训练时间、模型文件、预测文件与错误分析。模型比较必须使用相同验证集。
5. 使用验证集选择配置和停止轮数，再按固定方案用全部训练数据训练最终模型，预测测试集。融合与阈值选择也只能使用验证或折外预测。

实验记录建议字段为 `作业、实验编号、种子、划分文件、特征、模型参数、指标、耗时、结论、输出路径`。图片任务保存错误样例，回归保存残差图，分类保存混淆矩阵。不要根据测试集或排行榜反复倒推标签。

## 输出与报告

建议后续实现按 `src/hw1_person/` 等目录组织，实验产物放在 `outputs/<作业>/<实验编号>/`，其中保存配置、划分、模型、验证预测和提交文件。这些目录是建议结构，当前仓库尚未实现八套训练程序。

每份报告至少说明任务与数据、划分理由、基线、两项有解释的改进、消融结果、错误样例和复现命令。表格中的分数必须来自真实运行；没有提升的实验也应保留结论。小组成员可以分别负责数据与验证、模型、误差分析与提交，但必须共用划分和提交校验逻辑。

提交模板通常用 `pd.read_csv(...); sample[target] = predictions; sample.to_csv(..., index=False)` 填充。先按 ID 对齐，再赋值；模板中的默认预测值不是额外训练标签。文档去噪模板约 294 MB，应按其专用指南流式处理。

## 方法参考

- [scikit-learn 避免数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html)：预处理与模型放进同一 Pipeline，并限制在训练折拟合。
- [scikit-learn 交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html)：按任务选择分层、分组或时间划分。
- [scikit-learn 评价指标](https://scikit-learn.org/stable/modules/model_evaluation.html)：核对指标定义、标签顺序和分数方向。

原始数据依据为本地 `datasets/` 的 CSV、图片、manifest 与提交模板。数据不纳入 Git，完整字段、缺失计数和图片数量由 `data_profile.json` 保存，后续换数据版本时应重新统计。
