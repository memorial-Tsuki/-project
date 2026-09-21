"""使用已下载的真实行情快速检查 Backtrader 是否可以正常工作。

测试只读取 CSV 中第一只数据较完整的股票，不会生成随机价格，也不会连接网络。
"""

from pathlib import Path

import backtrader as bt
import pandas as pd


class MovingAverageStrategy(bt.Strategy):
    def __init__(self):
        short_average = bt.indicators.SMA(self.data.close, period=10)
        long_average = bt.indicators.SMA(self.data.close, period=30)
        self.cross = bt.indicators.CrossOver(short_average, long_average)

    def next(self):
        if not self.position and self.cross > 0:
            self.buy()
        elif self.position and self.cross < 0:
            self.close()


project_root = Path(__file__).resolve().parents[2]
data_file = (
    project_root
    / "data"
    / "tushare_csi300"
    / "csi300_qfq_close_2021_2025.csv"
)

if not data_file.is_file():
    raise FileNotFoundError(
        f"未找到真实行情文件：{data_file}\n"
        "请先运行 Alphalens 目录中的 Tushare 下载程序。"
    )

# CSV 是日期乘股票的宽表。选择有效值最多的一只股票，并删除它自己的缺失日。
prices = pd.read_csv(data_file, index_col=0, parse_dates=True)
prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
stock_code = str(prices.notna().sum().idxmax())
close = prices[stock_code].dropna().rename("close")

if len(close) < 60:
    raise ValueError("真实价格记录不足 60 天，无法完成均线测试。")

# Backtrader 的 PandasData 需要 OHLCV 列。基础测试只有收盘价，所以用真实收盘价
# 同时作为开、高、低、收；成交量设为 0。这里没有填造新的股票价格。
market = close.to_frame()
market["open"] = market["close"]
market["high"] = market["close"]
market["low"] = market["close"]
market["volume"] = 0.0
market["openinterest"] = 0.0

engine = bt.Cerebro()
engine.addstrategy(MovingAverageStrategy)
engine.adddata(bt.feeds.PandasData(dataname=market), name=stock_code)
engine.broker.setcash(100_000)
engine.broker.setcommission(commission=0.0003)
engine.addsizer(bt.sizers.FixedSize, stake=100)
engine.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

print("Backtrader 版本：", bt.__version__)
print("真实数据文件：", data_file)
print(f"测试股票：{stock_code}，交易日：{len(market)}")
print("初始资金：", engine.broker.getvalue())

result = engine.run()[0]

print("最终资金：", round(engine.broker.getvalue(), 2))
print("最大回撤：", round(result.analyzers.drawdown.get_analysis()["max"]["drawdown"], 2), "%")
print("Backtrader 测试成功")
