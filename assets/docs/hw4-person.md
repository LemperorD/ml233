# 个人作业第 4 题：内向／外向性格分类方法指导

课程依据：[作业说明 PDF](2026研究生机器学习大作业说明.pdf) 第 7 页；[课程邀请入口](https://www.kaggle.com/t/f9ddc5dbc15947fb82992d009d51492e) 来自 PDF 第 13 页。个人与团队各四选一；报告、截止时间及打包要求统一见[总览](README.md)。

利用七项行为属性预测 `Extrovert` 或 `Introvert`。主要难点是类别不平衡和多列缺失，先验证缺失处理与简单分类器，再比较树模型和阈值调整。

## 数据结构

本地目录 `datasets/ml-hw4-person/` 对应竞赛 `character`。训练集 14819 行，测试集 3705 行。输入为 `Time_spent_Alone`、`Stage_fear`、`Social_event_attendance`、`Going_outside`、`Drained_after_socializing`、`Friends_circle_size`、`Post_frequency`，输出为 `Personality`。

训练标签 `Extrovert` 有 10959 行，`Introvert` 有 3860 行；多数类占约 74%。始终预测多数类也有较高 Accuracy，必须用它作为对照，并检查少数类召回率。

`Stage_fear` 与 `Drained_after_socializing` 是 Yes/No 类别，其余五列是数值。训练缺失数分别为：独处时间 925、舞台恐惧 1506、社交活动 942、外出 1179、社交后疲惫 923、朋友圈 829、发帖频率 1006。测试集也存在缺失。

## 验证与指标

使用固定种子的五折 StratifiedKFold，或先分层留出 20% 跑通流程。全部清洗参数仅从训练折学习。先确认是否存在近重复行为组合；若后续数据发现重复人群记录，应按重复组隔离。

课程主指标已明确为 Accuracy，用它选择模型；Macro-F1、两类召回率和混淆矩阵用于辅助诊断。不能根据类别不平衡就直接把类别权重设为 balanced：它可能改善少数类召回，也可能降低官方关注的 Accuracy，应在同一划分比较。

## 最小基线

1. 多数类预测作为最低对照。
2. 用 ColumnTransformer 组织预处理：数值列中位数填充并加缺失标记，再标准化；Yes/No 列填 `Unknown` 并独热编码。
3. 将预处理和 LogisticRegression 放在同一个 Pipeline，比较正则化强度 `C=0.1,1,10`。
4. 明确标签编码，例如 `Introvert=1`、`Extrovert=0`，保存映射，预测后转回英文字符串。

不要把 Yes/No 缺失直接填为 No，因为“未提供”与“没有该行为”含义不同。也不要手工把独处时间大于某个值直接当成真实标签。

## 改进实验

比较 CatBoostClassifier 或其他梯度提升树，保留类别缺失信息；从浅树和早停开始。加入“缺失字段数”、独处时间与社交后疲惫的交互等特征，检查在各折是否稳定获益。

按 Accuracy 选模：先以 0.5 阈值为基准，再用折外预测选择能提高准确率的阈值。最终提交为字符串标签 `Extrovert` 或 `Introvert`，不能把内部预测概率直接写入提交列。

阈值选择和模型比较共用折外预测时，最终改进幅度还应在未参与选择的留出集或额外划分上复核。报告查看全部字段完整与缺失较多两组的性能，判断模型是否过度依赖缺失模式。

## 提交与报告

模板 `sample_submission.csv` 为 3705 行，列名严格使用 `id,Personality`。最终保存为 `submission.csv`，按 ID 对齐，只写 `Extrovert` 与 `Introvert`。检查预测类别比例显著偏离训练时的原因，不能仅为了匹配比例而强行更改预测。

报告给出多数类基线、逻辑回归、树模型、缺失指示与类别权重消融。讨论错误的行为组合时，围绕此数据集的预测关系，不把模型结果扩展成对真实人格的确定判断。

## 来源与规则入口

数据依据：[训练 CSV](../../datasets/ml-hw4-person/train.csv)、[模板](../../datasets/ml-hw4-person/sample_submission.csv) 与 [统计快照](data_profile.json)。[竞赛页面](https://www.kaggle.com/competitions/character) 用于核实补充规则；Accuracy 依据上述 PDF；验证与泄漏处理参考[总览](README.md#方法参考)。
