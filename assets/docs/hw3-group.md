# 团队作业第 3 题：人脸关键点定位方法指导

课程依据：[作业说明 PDF](2026研究生机器学习大作业说明.pdf) 第 11 页；[课程邀请入口](https://www.kaggle.com/t/b0527b03a6c0427eb161c02afb8ba528) 来自 PDF 第 13 页。个人与团队各四选一；报告、截止时间及打包要求统一见[总览](README.md)。

从灰度人脸图像预测十五个关键点的横纵坐标。重点是正确处理缺失标签、图像路径、几何增强和最终查表提交。先做坐标均值基线，再训练带标签掩码的小型 CNN。

## 数据与路径

目录 `datasets/ml-hw3-group/` 对应竞赛 `face-buaa`。训练集 5639 张，测试集 1410 张，全部图片为 96 × 96 灰度图。`train.csv` 包含 `ImageId,image_path` 和 30 个坐标字段；`keypoints.csv` 提供十五个点的名称及 x、y 列映射。

当前解压目录比 CSV 路径多一层：CSV 写 `train_images/face_00000000.png`，实际文件在 `train_images/train_images/face_00000000.png`；测试同理。读取时显式检查路径，不要在读取失败后用空白图片顶替。

```python
from pathlib import Path

root = Path("datasets/ml-hw3-group")
def resolve_image(relative_path):
    p = root / relative_path
    if p.is_file():
        return p
    rel = Path(relative_path)
    p = root / rel.parts[0] / rel
    if not p.is_file():
        raise FileNotFoundError(p)
    return p
```

## 缺失标签与验证

只有 1716 张训练图具备全部 30 个坐标。许多眼角、眉毛和嘴角字段缺失约 3800 行，不能把空坐标当成 0；也不宜只保留完整样本，丢弃大多数可用眼中心和鼻尖标注。

先检查完全相同或近重复人脸，按重复组隔离。再固定约 20% 验证，并尽量平衡完整标注与稀疏标注样本的比例。没有人物身份信息时，不声称已经完成身份级隔离。

课程明确对查找表内全部待预测坐标按原图像素单位计算 RMSE。本地在验证集有效坐标上累计计算像素 RMSE，以它选择模型，同时记录各关键点误差和标注数量。验证集缺失模式可能影响总体分数，应额外分别报告常见点与稀疏点的结果。

## 坐标与损失

先用训练折各坐标的有效值均值预测所有验证图片，作为最小基线。CNN 输入为 `[1,96,96]`，输出按固定列顺序排列的 30 维坐标。训练标注实测范围约为 0.687 至 95.809，可先按宽度 96 的连续坐标约定归一化：`u=(x-48)/48`，预测后用 `x=48*u+48` 还原，y 同理。

数组索引为 0 至 95，而标注含大于 95 的小数，因此不能把标签简单裁到整数索引范围。可视化核对坐标约定；几何增强库若采用像素中心而非连续边界坐标，必须显式统一变换，避免出现一个像素的系统偏移。

训练先保存有效掩码，再将无效目标替换为有限的占位值：

```python
mask = torch.isfinite(target)
safe_target = torch.where(mask, target, torch.zeros_like(target))
loss = (((pred - safe_target) ** 2) * mask).sum() / mask.sum().clamp_min(1)
```

直接计算含 NaN 的差再乘以零仍会传播 NaN。验证累计全部有效坐标的误差平方和与有效数量，最后统一开方，不平均不同标注数量批次的 RMSE。

## 模型与增强

先用三到四组卷积、归一化、池化和小型回归头，输入灰度归一化至 0 到 1。可从 Adam、学习率 `1e-3`、批量 64 开始，按验证 RMSE 早停。再比较残差骨干或关键点热图方案；使用预训练权重前核对规则。

按上述连续坐标约定，水平翻转使用 `x'=96-x`，归一化后即 `u'=-u`；若采用像素中心约定，则反射为 `95-x`，必须在整条预处理链中一致并用叠点图检查。翻转还必须交换左右眼、左右眉毛和左右嘴角对应列，连同缺失掩码一起交换。鼻尖和嘴唇中心保留语义，仅反射 x。旋转、缩放与裁剪必须将同一几何变换应用到坐标，并处理越界点。先只使用轻微亮度变化，再逐步加入几何增强，避免增强错误掩盖模型效果。

热图方案要忽略未标注点，保存下采样比例并正确还原位置。水平翻转测试增强同样需要反射坐标并交换点名后，才能平均预测。

## 查表提交

`IdLookupTable.csv` 有 21012 行，字段 `RowId,ImageId,FeatureName`；模板为 `sample_submission.csv`，字段 `RowId,Location`。它不是每张图固定提交全部 30 个坐标，不能直接把 1410 × 30 的预测展平。

最终文件命名为 `submission.csv`。先生成 `predictions[ImageId][FeatureName]`，再按查找表逐行提取相应值，最后按模板 `RowId` 对齐。检查所有图片和字段都能匹配，没有缺失或无穷值，坐标已从归一化尺度还原为原图像素。仅在确认官方边界后决定是否截断坐标。

## 小组交付与来源

2–3 人团队可由一人负责路径、掩码与验证，一人负责 CNN 和增强；第三人（如有）负责热图对照、可视化与查表提交，报告由全员核对。报告包含均值与 CNN 对照、完整与稀疏点误差、人脸上叠加真实点与预测点，以及翻转前后的可视化检查。

依据：[训练 CSV](../../datasets/ml-hw3-group/train.csv)、[点名映射](../../datasets/ml-hw3-group/keypoints.csv)、[提交查找表](../../datasets/ml-hw3-group/IdLookupTable.csv) 和 [统计快照](data_profile.json)。[竞赛页面](https://www.kaggle.com/competitions/face-buaa) 用于核实补充规则；像素单位和 RMSE 已由 PDF 明确；模型路线是根据本地标注结构提出的方案。
