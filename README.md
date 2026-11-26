# ml233

机器学习课程作业：四个个人作业与四个小组作业。

## 环境准备

- 注册 Kaggle 账户并配置 API Token，参考 [KaggleHub 认证说明](https://github.com/Kaggle/kagglehub#authenticate)。
- 安装下载依赖：`python3 -m pip install kagglehub`。
- 如需交互式配置认证，执行 `python3 -c "import kagglehub; kagglehub.login()"`。

## 下载数据集

```bash
python3 src/install_datasets.py
```

脚本按顺序下载八个数据集，保存到项目根目录下的 `datasets/`。仅下载部分作业时，注释其他作业对应的代码块。脚本默认使用已经配置的 Kaggle 凭据，不会每次提示登录。

若下载出现 `storage.googleapis.com` 连接超时，请检查终端的网络或代理设置，然后重新运行。不要将 API Token 提交到仓库。

## 作业方法指导

八份作业的任务分析、验证划分、模型路线与提交检查见 [方法指导总览](assets/docs/README.md)。本地数据统计见 [数据概况](assets/docs/data_profile.json)。指南是实现方案，当前尚未运行八个作业的训练与评分。
