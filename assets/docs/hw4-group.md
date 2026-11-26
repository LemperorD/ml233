# 小组作业四 文档图像去噪方法指导

把受噪声和背景干扰的灰度文档恢复成干净灰度图。目标图片含完整的 0 至 255 灰度值，因此先按连续像素回归实现；不要默认转成黑白二值分割。重点是原文档分组验证、成对训练和大规模像素提交。

## 数据与配对

目录为 `datasets/ml-hw4-group/`，竞赛标识 `document-buaa`。`train_manifest.csv` 有 232 行，对应 232 张 `train_noisy/` 输入和 232 张同名 `train_clean/` 目标；`test_manifest.csv` 有 56 行，对应 `test_noisy/`。

manifest 字段为 `image_id,height,width,variant`。训练包含 116 个文档的 `original` 和 `mixed` 两种输入；测试包含另外 28 个文档的两种输入。实际尺寸为宽 540、高 420 或 258，不能把高宽颠倒，也不能假定所有图都是同一高度。

核查发现，同一训练文档的 original 与 mixed 对应的 clean PNG 完全相同，116 对均如此。它们是同目标的两种退化输入，不应当作相互独立文档划入不同折。

## 验证与指标

从 `image_id` 去掉末尾 `_original` 或 `_mixed`，得到基础文档 ID；以它为组，留出约 20% 文档，或用 GroupKFold。必须先分文档，再裁训练块。若先裁块随机划分，会把同页字体、文字内容和背景泄漏到验证集。

本地在归一化到 0 至 1 的灰度上计算全像素 RMSE、MAE，附 PSNR 和视觉对比。官方像素尺度和指标需确认。RMSE 汇总时累计整批文档误差平方和与像素数量，再开方，避免短页与长页不恰当地等权。分别报告 original 和 mixed 的效果。

## 传统基线

1. 直接输出归一化噪声图，作为不处理的对照。
2. 比较轻微中值滤波、非局部均值去噪，或背景估计后校正。参数仅用训练文档与验证集选择。
3. 把噪声、恢复、clean 和误差图并排显示，放大细笔画、标点与字符交界处，检查是否去噪的同时损坏了文字。

传统滤波不一定能分离背景和字符，但能验证读取、归一化、指标和导出流程。阈值二值化可作为消融，不能直接当作连续灰度恢复的默认输出。

## 小型 U-Net 方案

用单通道输入与单通道输出的小型 U-Net，先采用 MSE 损失以匹配像素误差；也可比较残差学习，让网络预测噪声再从输入中扣除。训练前将输入和目标统一除以 255。最终输出限定在 0 至 1；若官方另有尺度，提交时统一转换。

初始可用 128 × 128 或 256 × 256 图块、批量 8 至 16、Adam 和约 `1e-3` 学习率，再按验证指标早停。这些是待实验的起始配置。采样同时覆盖文字密集区和背景，防止模型仅学会输出白底。

噪声与 clean 使用完全相同的几何变换；只对输入叠加合成退化时，必须保留目标不变，并确保其与实际噪声类型接近。不要将干净目标和噪声输入分别随机裁剪。先在单个训练块上检验能否过拟合，确认张量对齐后再完整训练。

## 整图推理与改进

模型步幅通常要求尺寸是某个倍数。推理时先填充到所需尺寸，恢复后裁掉填充，保持 manifest 的原始高宽。显存不足时用重叠滑窗，累计像素预测和权重再归一化；窗口边缘使用平滑权重，检查每个像素权重非零，避免拼接缝。

比较 MSE 与 L1、传统滤波与 U-Net、仅 original 训练与两类输入共同训练。若尝试结构损失，仍用官方对应像素指标选择模型。针对 mixed 和 original 的差异分析错误，不根据名字假设两种退化的具体生成算法。

## 1130 万行的提交

`sample_submission.csv` 约 294 MB，共 11301120 行，列为 `id,value`。此数等于测试 manifest 中所有 `height × width` 的总和。ID 样例为 `doc_034_original_1_1`，表示图像 ID、从 1 开始的行号、从 1 开始的列号；列先递增。

最稳妥的写法是分块读取模板，保留模板 ID 顺序，解析坐标并查预测图。假设 `pred_images` 已保存每张图的原尺寸二维、0 至 1 预测数组：

```python
import pandas as pd

source = "datasets/ml-hw4-group/sample_submission.csv"
first = True
with open("submission.csv", "w", newline="") as output:
    for chunk in pd.read_csv(source, chunksize=100_000):
        values = []
        for sid in chunk["id"]:
            image_id, row, col = sid.rsplit("_", 2)
            values.append(float(pred_images[image_id][int(row) - 1, int(col) - 1]))
        chunk["value"] = values
        chunk.to_csv(output, index=False, header=first)
        first = False
```

`rsplit("_",2)` 很关键，图像 ID 自身含下划线。上例是清晰的导出骨架，速度不足时按块向量化索引；不要一次构造一千多万个 Python 字典或读取整个模板到内存。检查输出行数、首末 ID、全部 ID 顺序、有限值与像素范围，确认没有转置或把 1 基坐标误作 0 基坐标。不能仅凭模板示例值为 1 就断言官方像素尺度，应核对 Evaluation。

## 小组交付与来源

建议分别负责文档分组与传统基线、成对图块与 U-Net、整图推理与误差可视化、流式导出与报告。报告重点解释分组验证为什么必要、文字笔画是否保留、两类退化的指标，以及大文件如何校验。

依据：[训练 manifest](../../datasets/ml-hw4-group/train_manifest.csv)、[测试 manifest](../../datasets/ml-hw4-group/test_manifest.csv)、本地成对图片与 [统计快照](data_profile.json)。[竞赛页面](https://www.kaggle.com/competitions/document-buaa) 用于确认尺度与规则；[scikit-image 非局部均值示例](https://scikit-image.org/docs/stable/auto_examples/filters/plot_nonlocal_means.html) 可用于传统基线。U-Net、采样和导出路线是针对本地文件结构制定的实现建议。
