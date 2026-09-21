"""使用真实价格 CSV 快速检查 Alphalens 是否可以正常工作。

本测试不会生成任何随机价格。为了缩短运行时间，它只截取真实数据中最近
260 个交易日，并使用数据完整度较高的前 50 只股票。
"""

from pathlib import Path

import alphalens as al
import pandas as pd


# 当前文件位于 labs/alphalens，parents[2] 是“计算金融”项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_PRICE_CSV = (
    PROJECT_ROOT
    / "data"
    / "tushare_csi300"
    / "csi300_qfq_close_2021_2025.csv"
)

if not REAL_PRICE_CSV.is_file():
    raise FileNotFoundError(
        f"未找到真实行情文件：{REAL_PRICE_CSV}\n"
        "请先运行“下载Tushare真实沪深300数据”。"
    )

# 第一列是日期，其余列是股票代码，单元格是真实前复权收盘价。
all_prices = pd.read_csv(REAL_PRICE_CSV, index_col=0, parse_dates=True)
all_prices = all_prices.sort_index().apply(pd.to_numeric, errors="coerce")

# 只截取真实数据的一个确定性子集来做快速测试，不产生或修改任何价格。
recent_prices = all_prices.tail(260)
minimum_valid_days = int(len(recent_prices) * 0.95)
prices = recent_prices.dropna(axis=1, thresh=minimum_valid_days).iloc[:, :50]

if prices.shape[1] < 5:
    raise ValueError("真实数据中满足完整度要求的股票太少，无法进行测试。")

# 测试因子使用真实价格计算过去 5 日收益，仍然不填补缺失价格。
factor_wide = prices.pct_change(5, fill_method=None).iloc[5:-5]
factor = factor_wide.stack()
factor.index.names = ["date", "asset"]

factor_data = al.utils.get_clean_factor_and_forward_returns(
    factor=factor,
    prices=prices,
    periods=(1, 5),
    quantiles=5,
    max_loss=1.0,
)
information_coefficient = al.performance.factor_information_coefficient(factor_data)

print(f"真实数据文件：{REAL_PRICE_CSV}")
print(f"测试价格表：{prices.shape[0]} 个交易日 × {prices.shape[1]} 只股票")
print("有效样本数：", len(factor_data))
print("平均 IC：")
print(information_coefficient.mean())
print("Alphalens 真实数据测试成功")
