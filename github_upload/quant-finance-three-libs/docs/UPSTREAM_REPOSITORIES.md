# 三个官方仓库和源码阅读入口

本项目只保存三套实验代码，不复制第三方仓库的全部源码。安装依赖后，Python 会从各自虚拟环境加载官方库。

## Alphalens

- 官方仓库：https://github.com/quantopian/alphalens
- 用途：清洗因子、计算未来收益、Rank IC、分组收益并画诊断图。
- 建议先看：`alphalens/utils.py`、`alphalens/performance.py`、`alphalens/plotting.py`、`alphalens/tears.py`。
- 本项目入口：`labs/alphalens/three_factors_analysis.py`。

如果 PyPI 安装失败，可在 Python 3.8 环境中使用官方源码安装：

```powershell
git clone https://github.com/quantopian/alphalens.git
.\.venv-alphalens\Scripts\python.exe -m pip install -e .\alphalens
```

## Backtrader

- 官方仓库：https://github.com/mementum/backtrader
- 用途：用事件驱动方式管理行情、策略、订单、持仓、资金和分析器。
- 建议先看：`backtrader/cerebro.py`、`backtrader/strategy.py`、`backtrader/feed.py`、`backtrader/broker.py`。
- 本项目入口：`labs/backtrader/three_factor_backtest.py`。

## Qlib

- 官方仓库：https://github.com/microsoft/qlib
- 用途：组织量化数据集、特征处理、机器学习训练、预测和实验记录。
- 建议先看：`qlib/data/dataset/`、`qlib/data/dataset/handler.py`、`qlib/contrib/model/gbdt.py`、`qlib/workflow/`。
- 本项目入口：`labs/qlib/three_factor_qlib.py`。

## 如何下载完整源码供阅读

完整源码不需要放进本项目仓库。可以在项目目录外单独执行：

```powershell
git clone https://github.com/quantopian/alphalens.git
git clone https://github.com/mementum/backtrader.git
git clone https://github.com/microsoft/qlib.git
```

`git clone` 会把远程仓库的代码、提交历史和分支信息复制到本地。以后进入对应目录执行 `git pull`，即可获取官方更新。

第三方源码遵循各自仓库中的许可证；如果以后复制或修改其源码并重新发布，需要同时保留相应许可证和版权说明。
