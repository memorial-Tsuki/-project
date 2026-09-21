"""使用 Backtrader 回测三个真实股票因子。

本文件和 Alphalens 示例使用完全相同的 Tushare 沪深300前复权收盘价，
并计算三个课件因子：

1. 20日动量：close_t / close_(t-20) - 1
2. 5日反转：-(close_t / close_(t-5) - 1)
3. 20日低波动：-std(daily_return, 20)

Alphalens回答“因子是否与未来收益有关”，Backtrader回答“按照因子实际调仓，
考虑资金和手续费之后会发生什么”。本示例每20个交易日调仓一次，等权持有
综合因子排名最高的20只股票。信号在当日收盘后产生，Backtrader默认让订单
在下一根K线执行，避免使用尚未知道的未来价格。

输出文件位于 output/backtrader_three_factors：
- backtest_metrics.csv：收益、波动、夏普和最大回撤；
- backtest_equity.csv：策略与等权基准的资金曲线；
- rebalance_holdings.csv：每次调仓选中的股票；
- backtest_result.png：可直接放入报告的图表。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, cast

import backtrader as bt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. 配置
# ---------------------------------------------------------------------------

# 当前文件位于 labs/backtrader，所以 parents[2] 是“计算金融”项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 把matplotlib缓存写到项目输出目录，避免用户主目录权限造成字体缓存警告。
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

# 正常从 VS Code 启动时使用 Qt5Agg，因此图表会显示为独立窗口。
# 自动化测试时把 BACKTRADER_SHOW_PLOTS 设为 0，改用不弹窗的 Agg 后端。
SHOW_PLOTS = os.environ.get("BACKTRADER_SHOW_PLOTS", "1") != "0"
AUTO_CLOSE_SECONDS = float(os.environ.get("BACKTRADER_AUTO_CLOSE_SECONDS", "0"))
import matplotlib

matplotlib.use("Qt5Agg" if SHOW_PLOTS else "Agg")
import matplotlib.pyplot as plt


# 如果以后换成你提供的另一份真实数据，可以设置 BACKTRADER_PRICE_CSV；
# 没有设置时继续使用当前 Tushare 沪深300文件。
PRICE_CSV_TEXT = os.environ.get("BACKTRADER_PRICE_CSV", "").strip()
DEFAULT_PRICE_CSV = (
    PROJECT_ROOT
    / "data"
    / "tushare_csi300"
    / "csi300_qfq_close_2021_2025.csv"
)
PRICE_CSV = Path(PRICE_CSV_TEXT) if PRICE_CSV_TEXT else DEFAULT_PRICE_CSV
OUTPUT_DIR = PROJECT_ROOT / "output" / "backtrader_three_factors"

# 回测参数集中放在这里，便于初学者做敏感性实验。
INITIAL_CASH = 1_000_000.0       # 初始资金100万元
TOP_N = 20                       # 每次持有综合排名最高的20只股票
REBALANCE_EVERY = 20             # 每20个交易日调仓一次
TARGET_GROSS_EXPOSURE = 0.90     # 最多使用90%资金，给手续费和价格波动留余量
COMMISSION_RATE = 0.0003         # 双边佣金示例：万分之三
WARMUP_DAYS = 20                 # 最长因子需要20天历史


# ---------------------------------------------------------------------------
# 2. 读取真实价格并计算三个因子
# ---------------------------------------------------------------------------

def load_real_prices(csv_path: Path) -> pd.DataFrame:
    """读取Tushare生成的真实价格宽表并完成基础检查。

    CSV的行是日期，列是股票代码，值是前复权收盘价。函数不会生成模拟数据；
    如果文件不存在就直接报错，提醒先运行下载程序。
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"未找到真实行情文件：{csv_path}\n"
            "请先运行 Alphalens 目录中的 download_tushare_csi300.py。"
        )

    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    prices.index.name = "date"
    prices = cast(
        pd.DataFrame,
        prices.sort_index().apply(pd.to_numeric, errors="coerce"),
    )

    if prices.empty:
        raise ValueError("真实价格表为空。")
    if prices.index.has_duplicates:
        raise ValueError("真实价格表存在重复日期。")
    if (prices <= 0).any().any():
        raise ValueError("价格表中出现了小于等于0的价格。")

    missing_ratio = prices.isna().sum().sum() / prices.size
    print(f"数据来源：Tushare真实沪深300前复权收盘价 {csv_path}")
    print(f"价格表：{prices.shape[0]}个交易日，{prices.shape[1]}只股票")
    print(f"价格缺失率：{missing_ratio:.2%}")
    return prices


def calculate_three_factors(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """按照课件公式计算三个因子，方向统一为“数值越高越好”。"""
    # fill_method=None很重要：遇到停牌或缺失价格时不要偷偷向前填充。
    daily_returns = prices.pct_change(fill_method=None)

    factors = {
        # 过去20日涨得越多，动量因子越高。
        "momentum_20d": prices / prices.shift(20) - 1,
        # 过去5日涨幅取负号，近期跌得越多，反转因子越高。
        "reversal_5d": -(prices / prices.shift(5) - 1),
        # 日收益率20日标准差取负号，波动越小，因子越高。
        "low_volatility_20d": -daily_returns.rolling(
            20, min_periods=20
        ).std(),
    }
    return factors


def build_composite_score(
    factors: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """把量纲不同的三个因子转换成每日百分位排名，再等权平均。

    例如某只股票动量排名为0.90，反转排名为0.40，低波动排名为0.80，
    综合分数就是 (0.90 + 0.40 + 0.80) / 3 = 0.70。
    """
    percentile_ranks = [
        factor.rank(axis=1, method="average", pct=True)
        for factor in factors.values()
    ]
    if not percentile_ranks:
        raise ValueError("至少需要一个因子才能计算综合分数。")

    # 不直接使用 sum(DataFrame列表)：pandas 的类型声明认为 sum 可能从数字0
    # 开始，Pylance因而误判返回值可能是float。转成三维数组后按因子轴求均值，
    # 再恢复原来的日期和股票标签，计算含义与逐表相加后除以因子数量完全相同。
    rank_cube = np.stack(
        [rank.to_numpy(dtype=float) for rank in percentile_ranks], axis=0
    )
    return pd.DataFrame(
        rank_cube.mean(axis=0),
        index=percentile_ranks[0].index,
        columns=percentile_ranks[0].columns,
    )


# ---------------------------------------------------------------------------
# 3. 把 pandas 数据交给 Backtrader
# ---------------------------------------------------------------------------

class PandasThreeFactorData(bt.feeds.PandasData):
    """在Backtrader标准OHLC数据上增加四条因子线。

    Backtrader把每个字段称为一条 line。这样策略在每个交易日可以像读取
    data.close[0] 一样读取 data.composite[0]。
    """

    lines = (
        "momentum_20d",
        "reversal_5d",
        "low_volatility_20d",
        "composite",
    )
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", "openinterest"),
        ("momentum_20d", "momentum_20d"),
        ("reversal_5d", "reversal_5d"),
        ("low_volatility_20d", "low_volatility_20d"),
        ("composite", "composite"),
    )


def add_stock_feeds(
    cerebro: bt.Cerebro,
    prices: pd.DataFrame,
    factors: dict[str, pd.DataFrame],
    composite: pd.DataFrame,
) -> None:
    """为每只股票创建一条Backtrader数据源。

    Tushare宽表只有收盘价。为了演示日频组合回测，这里把同一天的开、高、低
    都设为收盘价；因此订单的下一日执行价等于下一交易日收盘价。生产研究应当
    继续下载真实开盘价、最高价、最低价和成交量。
    """
    # Backtrader通过元类动态接收dataname、name等关键字；第三方类型信息没有
    # 声明这些参数。只把构造器视为Any，不影响创建出的真实数据对象。
    data_feed_class = cast(Any, PandasThreeFactorData)

    for ts_code in prices.columns:
        # 删除该股票缺失的日期，而不是把缺失价格填成0。
        close = prices[ts_code].dropna()
        if len(close) <= WARMUP_DAYS + 5:
            continue

        frame = pd.DataFrame(index=close.index)
        frame["open"] = close
        frame["high"] = close
        frame["low"] = close
        frame["close"] = close
        frame["volume"] = 0.0
        frame["openinterest"] = 0.0
        frame["momentum_20d"] = factors["momentum_20d"][ts_code].reindex(
            close.index
        )
        frame["reversal_5d"] = factors["reversal_5d"][ts_code].reindex(
            close.index
        )
        frame["low_volatility_20d"] = factors[
            "low_volatility_20d"
        ][ts_code].reindex(close.index)
        frame["composite"] = composite[ts_code].reindex(close.index)

        data = data_feed_class(dataname=frame, name=str(ts_code))
        cerebro.adddata(data)


# ---------------------------------------------------------------------------
# 4. 三因子组合策略
# ---------------------------------------------------------------------------

class ThreeFactorPortfolio(bt.Strategy):
    """每20个交易日等权持有综合因子最高的20只股票。"""

    params = dict(
        top_n=TOP_N,
        rebalance_every=REBALANCE_EVERY,
        target_gross_exposure=TARGET_GROSS_EXPOSURE,
    )

    def __init__(self):
        # rebalance_records最终会保存为CSV，便于检查每次到底买了什么。
        self.rebalance_records: list[dict[str, object]] = []
        self.failed_orders = 0

    def _data_has_current_bar(self, data, current_date) -> bool:
        """判断股票在当前主交易日是否真的有行情，而不是停留在旧K线。"""
        return len(data) > 0 and data.datetime.date(0) == current_date

    def prenext(self):
        """在并非所有股票都已上市时，也按已有股票正常执行策略。

        Backtrader面对300条起始日期不同的数据时，会把“所有数据都开始前”的
        阶段交给 prenext。若不实现本方法，策略会错误地等到最后一只股票上市
        后才开始。调用 next 后，我们仍会逐只检查股票当天是否真的有行情。
        """
        self.next()

    def next(self):
        """Backtrader每推进一个交易日调用一次。"""
        # 第一条数据是主时钟。我们在加入数据时保留了权重表的顺序，第一只股票
        # 拥有完整交易日期，因此适合作为组合的日历。
        current_date = self.datas[0].datetime.date(0)

        # 前20个交易日没有足够历史计算因子；之后每20天调仓一次。
        if len(self) <= WARMUP_DAYS:
            return
        if (len(self) - WARMUP_DAYS - 1) % self.p.rebalance_every != 0:
            return

        candidates = []
        for data in self.datas:
            if not self._data_has_current_bar(data, current_date):
                continue
            score = float(data.composite[0])
            price = float(data.close[0])
            if math.isfinite(score) and math.isfinite(price) and price > 0:
                candidates.append((score, data))

        # 如果有效股票少于目标持仓数，就暂时不调仓。
        if len(candidates) < self.p.top_n:
            return

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[: self.p.top_n]
        selected_datas = {data for _, data in selected}

        # 先把不再入选的股票目标仓位调为0。Backtrader会在下一根K线执行。
        for data in self.datas:
            if self.getposition(data).size != 0 and data not in selected_datas:
                self.order_target_percent(data=data, target=0.0)

        # 剩余90%资金在入选股票之间等权分配。
        target_weight = self.p.target_gross_exposure / len(selected)
        for score, data in selected:
            self.order_target_percent(data=data, target=target_weight)
            self.rebalance_records.append(
                {
                    "signal_date": current_date.isoformat(),
                    "ts_code": data._name,
                    "composite_score": score,
                    "target_weight": target_weight,
                }
            )

        print(
            f"调仓信号 {current_date}：有效股票{len(candidates)}只，"
            f"买入前{len(selected)}只"
        )

    def notify_order(self, order):
        """记录被拒绝、保证金不足或取消的订单。"""
        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.failed_orders += 1
            print(
                f"订单未成交：{order.data._name}，"
                f"状态={order.getstatusname()}"
            )


# ---------------------------------------------------------------------------
# 5. 结果计算、保存与绘图
# ---------------------------------------------------------------------------

def build_equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
    """构造简单等权基准，作为策略资金曲线的参照。"""
    returns = prices.pct_change(fill_method=None)
    benchmark_daily_return = returns.mean(axis=1, skipna=True).fillna(0.0)
    return INITIAL_CASH * (1.0 + benchmark_daily_return).cumprod()


def save_results(
    strategy: ThreeFactorPortfolio,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从Backtrader分析器取回结果并保存CSV。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily_returns = pd.Series(
        strategy.analyzers.daily_returns.get_analysis(), dtype=float
    )
    daily_returns.index = pd.to_datetime(daily_returns.index)
    daily_returns = daily_returns.sort_index()
    strategy_equity = INITIAL_CASH * (1.0 + daily_returns).cumprod()

    benchmark = build_equal_weight_benchmark(prices).reindex(
        strategy_equity.index
    ).ffill()
    equity = pd.DataFrame(
        {
            "strategy_equity": strategy_equity,
            "equal_weight_benchmark": benchmark,
        }
    )
    equity.index.name = "date"
    equity.to_csv(OUTPUT_DIR / "backtest_equity.csv", encoding="utf-8-sig")

    # 从每日收益自行计算年化指标，定义清晰、便于学习和复核。
    number_of_days = max(len(daily_returns), 1)
    final_value = float(strategy.broker.getvalue())
    total_return = final_value / INITIAL_CASH - 1.0
    annualized_return = (final_value / INITIAL_CASH) ** (
        252.0 / number_of_days
    ) - 1.0
    annualized_volatility = daily_returns.std(ddof=1) * np.sqrt(252.0)
    sharpe = (
        daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252.0)
        if daily_returns.std(ddof=1) > 0
        else np.nan
    )
    max_drawdown = strategy.analyzers.drawdown.get_analysis()["max"][
        "drawdown"
    ] / 100.0

    metrics = pd.DataFrame(
        [
            {
                "initial_cash": INITIAL_CASH,
                "final_value": final_value,
                "total_return": total_return,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratio_rf0": sharpe,
                "max_drawdown": max_drawdown,
                "failed_orders": strategy.failed_orders,
                "stocks_per_rebalance": TOP_N,
                "rebalance_every_days": REBALANCE_EVERY,
                "commission_rate": COMMISSION_RATE,
            }
        ]
    )
    metrics.to_csv(
        OUTPUT_DIR / "backtest_metrics.csv", index=False, encoding="utf-8-sig"
    )

    holdings = pd.DataFrame(strategy.rebalance_records)
    holdings.to_csv(
        OUTPUT_DIR / "rebalance_holdings.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return equity, metrics


def plot_results(equity: pd.DataFrame) -> None:
    """绘制资金曲线和策略回撤，并保存PNG。"""
    normalized = equity / equity.iloc[0]
    drawdown = equity["strategy_equity"] / equity[
        "strategy_equity"
    ].cummax() - 1.0

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    normalized.plot(ax=axes[0], linewidth=1.4)
    axes[0].set_title("Three-factor Backtrader equity curve")
    axes[0].set_ylabel("Growth of 1 unit")
    axes[0].grid(alpha=0.25)

    axes[1].fill_between(
        drawdown.index,
        drawdown.to_numpy(dtype=float) * 100.0,
        0,
        color="#c44e52",
        alpha=0.65,
    )
    axes[1].set_title("Strategy drawdown")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "backtest_result.png", dpi=160)

    if SHOW_PLOTS:
        print("Backtrader图形窗口已经打开；关闭窗口后程序才会结束。")
        if AUTO_CLOSE_SECONDS > 0:
            timer = figure.canvas.new_timer(
                interval=int(AUTO_CLOSE_SECONDS * 1000)
            )
            timer.add_callback(plt.close, figure)
            timer.start()
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    """按“真实数据 → 因子 → 回测 → 保存 → 绘图”的顺序执行。"""
    prices = load_real_prices(PRICE_CSV)
    factors = calculate_three_factors(prices)
    composite = build_composite_score(factors)

    # Cerebro同样使用元类动态接收stdstats参数，运行时合法但类型信息未声明。
    cerebro_class = cast(Any, bt.Cerebro)
    cerebro = cerebro_class(stdstats=False)
    cerebro.addstrategy(ThreeFactorPortfolio)
    add_stock_feeds(cerebro, prices, factors, composite)

    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION_RATE)
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        # Days是Backtrader运行时动态添加的枚举属性。
        timeframe=getattr(bt.TimeFrame, "Days"),
        _name="daily_returns",
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    print(f"Backtrader版本：{bt.__version__}")
    print(f"初始资金：{INITIAL_CASH:,.2f}")
    strategy = cerebro.run(runonce=True, preload=True)[0]

    equity, metrics = save_results(strategy, prices)
    print("\n回测核心指标：")
    print(metrics.round(6).to_string(index=False))
    print(f"\n结果已保存到：{OUTPUT_DIR}")
    print("确认：本回测使用Tushare真实沪深300前复权收盘价，不使用随机行情。")
    plot_results(equity)


if __name__ == "__main__":
    main()
