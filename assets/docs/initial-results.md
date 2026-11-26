# 初版本地验证记录

日期：2026-11-26。八项均使用完整本地训练数据进行验证与全量重训，生成测试提交；图像网络仅做一轮流程检查，尚未调优。下表是本地验证结果，不是 Kaggle 成绩；不同任务的分数不可横向比较。

| 作业 | 实际运行模型与设置 | 验证主指标 | 简单基线同指标 | 提交行数 |
| --- | --- | --- | --- | --- |
| 个人 1：葡萄酒 | ridge，默认参数 | rmse=0.629397 | 0.804657 | 480 |
| 个人 2：美食 | cnn，1 轮、64×64 | accuracy=0.142077 | 0.109290 | 608 |
| 个人 3：背包 | ridge，默认参数 | rmse=38.949586 | 38.971285 | 60000 |
| 个人 4：性格 | logistic，默认参数 | accuracy=0.977058 | 0.739541 | 3705 |
| 团队 1：单车 | gradient_boosting，40 次提升 | rmsle=0.314444 | 1.557115 | 2286 |
| 团队 2：作者 | logistic，默认参数 | log_loss=0.513929 | 1.088672 | 4895 |
| 团队 3：关键点 | cnn，1 轮 | rmse=4.191220 | 4.126985 | 21012 |
| 团队 4：去噪 | unet，1 轮、64×64、每图 1 块 | rmse=0.102059 | 0.176682 | 11301120 |

葡萄酒为重复特征分组五折折外指标。背包按重复特征组留出；性格和作者使用分层留出；单车按月内 1–11 日训练、12–15 日验证；人脸重复图分组；去噪按原文档分组。种子均为 42，运行设备为 CPU。各模型使用训练折拟合预处理；验证模型与全量模型分别保存。

基线分别为训练折均值、训练折多数类、训练折价格均值、训练折多数类、训练折计数均值、训练折作者先验概率、训练折各有效坐标均值、直接输出噪声图。人脸的一轮检查未优于坐标均值基线。去噪 U-Net 采用零初始化残差输出头，一轮验证 RMSE 为 0.102059，低于原图基线 0.176682；仍需继续训练并检查泛化。美食另有颜色直方图基线，同划分 Accuracy=0.273224、Macro-F1=0.261148。

实际配置、依赖版本和详细指标保存在 [initial-results.json](initial-results.json)。本地实验目录为 `outputs/<作业>/initial_check/`，模型与大规模提交文件不纳入 Git。

## 复现本次设置

使用根 README 的环境准备命令，在项目根目录执行；目录已存在时更换实验名：

```bash
python -m src.hw1_person.train --output-dir outputs/hw1_person/initial_check
python -m src.hw2_person.train --epochs 1 --image-size 64 --device cpu --output-dir outputs/hw2_person/initial_check
python -m src.hw3_person.train --output-dir outputs/hw3_person/initial_check
python -m src.hw4_person.train --output-dir outputs/hw4_person/initial_check
python -m src.hw1_group.train --trees 40 --output-dir outputs/hw1_group/initial_check
python -m src.hw2_group.train --output-dir outputs/hw2_group/initial_check
python -m src.hw3_group.train --epochs 1 --device cpu --output-dir outputs/hw3_group/initial_check
python -m src.hw4_group.train --epochs 1 --patch-size 64 --patches-per-image 1 --device cpu --output-dir outputs/hw4_group/initial_check
python -m src.hw4_group.predict --run-dir outputs/hw4_group/initial_check --device cpu
```

自动化验证：`python -m unittest discover -s tests -v`。检查使用临时数据，覆盖全部八项入口和所有可选模型，同时检查重复分组、日期隔离、缺失坐标、成对裁剪、模板顺序及跨块像素重复/遗漏检测。
