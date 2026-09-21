"""用 Alphalens 分析课件中的三个示例因子（初学者教学版）。

三个因子：
1. 20 日动量：close_t / close_(t-20) - 1
2. 5 日反转：-(close_t / close_(t-5) - 1)
3. 20 日低波动：-std(return_(t-19), ..., return_t)

整个程序的数据流如下：

    价格数据
       ↓
    根据课件公式计算三个因子
       ↓
    转成 Alphalens 要求的 (date, asset) 双层索引
       ↓
    计算未来 1 日、5 日收益
       ↓
    计算 Rank IC、五分组收益并画图
       ↓
    保存 CSV 和 PNG 文件

当前脚本只读取用户提供或 Tushare 下载的沪深300真实前复权收盘价。
如果真实数据文件不存在，程序会立即报错；本文件不包含模拟行情生成器。
"""

# pathlib.Path 用来安全地拼接 Windows 文件路径，避免手写很多反斜杠。
from pathlib import Path
# os 用来读取 VS Code 启动配置传入的环境变量。
import os
# warnings 用来隐藏 Alphalens 旧版本依赖产生、但不影响运行的警告。
import warnings

# al 是 Alphalens 的简称，后面会用到 al.utils、al.performance、al.plotting。
import alphalens as al
# matplotlib 是绘图库，Qt5Agg 后端负责弹出独立图形窗口。
import matplotlib
# pandas 负责表格、日期索引、收益率和滚动窗口计算。
import pandas as pd


# 使用 Qt 图形后端。正常运行时会弹出图表窗口。
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. 配置
# ---------------------------------------------------------------------------

# __file__ 是当前脚本路径；parents[2] 回到“计算金融”项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 默认真实数据文件：由 download_tushare_csi300.py 生成。
DEFAULT_REAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "tushare_csi300"
    / "csi300_qfq_close_2021_2025.csv"
)

# 可通过环境变量指定另一份真实价格 CSV；没有指定时使用项目默认真实数据。
CSV_PATH_TEXT = os.environ.get("ALPHALENS_PRICE_CSV", "").strip()
CSV_PATH = Path(CSV_PATH_TEXT) if CSV_PATH_TEXT else DEFAULT_REAL_CSV

# 真实 CSV 格式必须是：第一列为日期，其余列为股票代码，单元格为复权收盘价。
# 示例：
# date,000001.SZ,000002.SZ,600000.SH
# 2024-01-02,10.23,9.81,8.54
# 2024-01-03,10.35,9.76,8.61

# 研究因子对未来 1 个和 5 个交易日收益的预测能力。
FORWARD_PERIODS = (1, 5)
# 每天按照因子值从低到高，把股票分成 5 组。
QUANTILES = 5
# 最长因子需要过去 20 天数据，因此三个因子统一从第 21 天开始比较。
COMMON_WARMUP_DAYS = 20
# 下面三个设置主要由 .vscode/launch.json 自动传入，初学时不用修改。
# SHOW_PLOTS=True 表示计算完成后显示图形窗口。
SHOW_PLOTS = os.environ.get("ALPHALENS_SHOW_PLOTS", "1") != "0"
# 自动测试时可以让窗口几秒后关闭；正常运行时为 0，窗口不会自动关闭。
AUTO_CLOSE_SECONDS = float(os.environ.get("ALPHALENS_AUTO_CLOSE_SECONDS", "0"))
# 如果这里收到某个因子名，就打开该因子的 Alphalens 原生 tear sheet。
NATIVE_TEAR_SHEET_FACTOR = os.environ.get(
    "ALPHALENS_NATIVE_TEAR_SHEET_FACTOR", ""
).strip()

# 所有结果统一写入这个目录，避免散落在源码文件夹里。
OUTPUT_DIR = PROJECT_ROOT / "output" / "alphalens_three_factors"


def load_real_prices(csv_path):
    """读取并整理用户提供的真实价格 CSV。

    参数
    ----
    csv_path : Path 或字符串
        真实价格 CSV 的路径，不允许省略。

    返回
    ----
    prices : pandas.DataFrame
        行索引是交易日期，列名是股票代码，数值是复权收盘价。
        这就是 Alphalens 对 prices 参数要求的“宽表”结构。
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            "未找到真实行情文件："
            f"{csv_path.resolve()}\n"
            "请先运行“下载Tushare真实沪深300数据”，或设置 "
            "ALPHALENS_PRICE_CSV 指向你提供的真实 CSV。"
        )
    # index_col=0：把 CSV 第一列当作行索引。
    # parse_dates=True：把日期文本转换为 pandas 日期类型。
    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    prices.index.name = "date"
    # 按日期升序排列，并把不能转成数字的内容变为 NaN，方便后面检查。
    prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    return prices


def validate_prices(prices):
    """完成课件要求的最小数据质量检查。

    数据问题如果拖到因子分析阶段才暴露，会很难定位。因此在研究开始前，
    先检查空表、日期格式、重复日期、排序、重复股票代码和非法价格。
    """
    if prices.empty:
        raise ValueError("价格表为空。")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("价格表索引必须是日期。")
    if prices.index.has_duplicates:
        raise ValueError("价格表存在重复日期。")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("价格表日期必须按升序排列。")
    if prices.columns.has_duplicates:
        raise ValueError("价格表存在重复股票代码。")
    if (prices <= 0).any().any():
        raise ValueError("价格必须大于 0。")

    # prices.size 是全部单元格数量；缺失值数量除以它就是总体缺失率。
    missing_ratio = float(prices.isna().sum().sum() / prices.size)
    print(f"价格表：{prices.shape[0]} 个交易日，{prices.shape[1]} 只资产")
    print(f"价格缺失率：{missing_ratio:.2%}")


def calculate_factors(prices):
    """按照课件公式计算三个因子，全部统一为“分数越高越好”。

    每个返回值仍然是宽表：行是日期，列是股票代码。
    前 5 天或 20 天因为历史数据不足会出现 NaN，这是正常现象。
    """
    # pct_change() 计算相邻两个交易日的简单收益率：P_t / P_(t-1) - 1。
    returns = prices.pct_change()

    factors = {
        # 20 日动量：今天价格相对 20 个交易日前涨得越多，分数越高。
        "momentum_20d": prices / prices.shift(20) - 1,
        # 5 日反转：先算过去 5 日涨幅，再取负号。
        # 最近跌得越多，负号之后的分数越高，表达“短期下跌后可能反弹”。
        "reversal_5d": -(prices / prices.shift(5) - 1),
        # 20 日低波动：rolling(20).std() 计算过去 20 日收益率标准差。
        # 取负号后，波动率越低的股票分数越高。
        "low_volatility_20d": -returns.rolling(20, min_periods=20).std(),
    }
    return factors


def wide_factor_to_series(factor_wide):
    """把“日期×股票”宽表转成 Alphalens 要求的双层索引 Series。

    宽表原来类似：

                 stock_A  stock_B
        2024-01      0.10    -0.05

    stack() 后变成：

        date       asset
        2024-01    stock_A     0.10
                   stock_B    -0.05

    Alphalens 要求索引层名称必须能识别为 date 和 asset。
    """
    # 最长未来收益期为 5 天，所以最后 5 天没有足够的未来价格，提前去掉。
    max_forward_period = max(FORWARD_PERIODS)
    # 三个因子统一使用第 21 个交易日起的样本，保证横向比较公平。
    factor_wide = factor_wide.iloc[COMMON_WARMUP_DAYS:-max_forward_period]
    # stack() 把股票列折叠到第二层索引；dropna() 去掉历史不足产生的空值。
    factor = factor_wide.stack().dropna()
    factor.index.names = ["date", "asset"]
    factor.name = "factor"
    return factor


def analyze_factor(name, factor, prices):
    """清洗一个因子，并计算 Rank IC 与五分组收益。

    name 是因子名称；factor 是双层索引因子序列；prices 是价格宽表。
    返回汇总字典、每日 IC、分组收益和 Alphalens 清洗后的标准数据表。
    """
    # 这是 Alphalens 最核心的入口函数。它会完成：
    # 1. 对齐因子日期、股票和价格；
    # 2. 计算未来 1 日与 5 日收益；
    # 3. 每天按因子值划分为 5 个分位数组；
    # 4. 删除无法匹配的数据并报告损失比例。
    factor_data = al.utils.get_clean_factor_and_forward_returns(
        factor=factor,
        prices=prices,
        periods=FORWARD_PERIODS,
        quantiles=QUANTILES,
        # 如果清洗导致超过 35% 的因子样本丢失，就报错提醒检查数据。
        max_loss=0.35,
    )

    # Rank IC 是每日横截面上“因子排名”和“未来收益排名”的 Spearman 相关系数。
    # IC > 0 通常表示高因子值股票的未来收益更高；接近 0 表示关系很弱。
    rank_ic = al.performance.factor_information_coefficient(factor_data)
    # 按 1 到 5 分组计算未来平均收益。
    # demeaned=False 表示输出每组的原始平均收益，不先减去全市场平均收益。
    mean_returns, _ = al.performance.mean_return_by_quantile(
        factor_data,
        demeaned=False,
    )
    # 第 5 组减第 1 组，用来观察高分组和低分组之间有没有经济差异。
    spread = mean_returns.loc[QUANTILES] - mean_returns.loc[1]

    # 以下两张图会单独保存到硬盘，方便写报告时直接引用。
    # 乘以 100 是把小数收益率转换成百分数，例如 0.01 变成 1%。
    quantile_percent = mean_returns * 100
    axes = quantile_percent.plot(
        kind="bar",
        figsize=(8, 4.5),
        title=f"{name}: mean forward return by quantile",
    )
    axes.set_xlabel("Factor quantile (1=lowest, 5=highest)")
    axes.set_ylabel("Mean forward return (%)")
    axes.figure.tight_layout()
    axes.figure.savefig(OUTPUT_DIR / f"{name}_quantile_returns.png", dpi=150)
    plt.close(axes.figure)

    # 每日 IC 波动较大，用 20 日移动平均观察方向是否持续。
    axes = rank_ic.rolling(20, min_periods=5).mean().plot(
        figsize=(8, 4.5),
        title=f"{name}: 20-day rolling Rank IC",
    )
    axes.axhline(0, color="black", linewidth=0.8)
    axes.set_xlabel("Date")
    axes.set_ylabel("Rank IC")
    axes.figure.tight_layout()
    axes.figure.savefig(OUTPUT_DIR / f"{name}_rolling_rank_ic.png", dpi=150)
    plt.close(axes.figure)

    # 保存中间结果。以后写报告、检查异常或复现实验时无需重新计算。
    factor_data.to_csv(OUTPUT_DIR / f"{name}_clean_factor_data.csv")
    rank_ic.to_csv(OUTPUT_DIR / f"{name}_daily_rank_ic.csv")
    mean_returns.to_csv(OUTPUT_DIR / f"{name}_quantile_returns.csv")

    # 整理成一行汇总结果，最后三个因子会拼成 factor_summary.csv。
    result = {
        "factor": name,
        "observations": len(factor_data),
        "rank_ic_1d": rank_ic["1D"].mean(),
        "rank_ic_1d_std": rank_ic["1D"].std(),
        "rank_ic_5d": rank_ic["5D"].mean(),
        "rank_ic_5d_std": rank_ic["5D"].std(),
        "q5_minus_q1_1d": spread["1D"],
        "q5_minus_q1_5d": spread["5D"],
    }
    return result, rank_ic, mean_returns, factor_data


def create_combined_chart(plot_results):
    """调用 Alphalens 原生 plotting，把三个因子放进一个 3×3 弹窗。

    第一行：五分组收益；第二行：1 日 Rank IC；第三行：5 日 Rank IC。
    每一列对应一个因子，便于在同一尺度下横向比较。
    """
    # axes 是 3 行 × 3 列的坐标轴数组。
    figure, axes = plt.subplots(3, 3, figsize=(16, 12))

    for column, (name, rank_ic, mean_returns) in enumerate(plot_results):
        # 直接调用 Alphalens plotting.py 内置的分组收益柱状图。
        al.plotting.plot_quantile_returns_bar(
            mean_returns,
            by_group=False,
            ax=axes[0, column],
        )
        axes[0, column].set_title(f"{name}: quantile returns")
        axes[0, column].tick_params(axis="x", rotation=0)

        # plot_ic_ts 会同时画每日 IC 和约一个月的移动平均线。
        al.plotting.plot_ic_ts(rank_ic[["1D"]], ax=[axes[1, column]])
        axes[1, column].set_title(f"{name}: 1D Rank IC")

        al.plotting.plot_ic_ts(rank_ic[["5D"]], ax=[axes[2, column]])
        axes[2, column].set_title(f"{name}: 5D Rank IC")

    figure.suptitle("Alphalens native plotting for three factors", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(OUTPUT_DIR / "three_factors_diagnostics.png", dpi=150)
    return figure


def main():
    """按照“准备数据 → 计算因子 → 分析 → 保存 → 展示”的顺序运行。"""
    # parents=True 会连同缺失的上层目录一起创建；exist_ok=True 表示已存在也不报错。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 这里只接受真实 CSV；文件缺失时 load_real_prices 会立即报错。
    prices = load_real_prices(CSV_PATH)
    print(f"本次数据来源：真实价格 CSV：{CSV_PATH}")
    validate_prices(prices)
    # 保存本次实际使用的价格，保证以后能够复现结果。
    prices.to_csv(OUTPUT_DIR / "prices_used.csv")

    # factors 是字典：键是因子名，值是“日期×股票”的因子宽表。
    factors = calculate_factors(prices)
    # 下面四个容器分别收集长表、汇总指标、绘图数据和清洗后数据。
    long_factor_tables = []
    results = []
    plot_results = []
    clean_factor_data = {}

    # 循环三次，每次处理一个因子。
    for name, factor_wide in factors.items():
        factor = wide_factor_to_series(factor_wide)

        # 课件要求的统一长表格式：date, asset, factor, value。
        long_table = factor.rename("value").reset_index()
        long_table.insert(2, "factor", name)
        long_factor_tables.append(long_table)

        # 执行 Alphalens 清洗、统计、绘图保存和 CSV 输出。
        result, rank_ic, mean_returns, factor_data = analyze_factor(
            name, factor, prices
        )
        results.append(result)
        plot_results.append((name, rank_ic, mean_returns))
        clean_factor_data[name] = factor_data

    # 把三个因子拼成课件要求的 date, asset, factor, value 长表。
    all_factors = pd.concat(long_factor_tables, ignore_index=True)
    all_factors.to_csv(OUTPUT_DIR / "three_factors_long_format.csv", index=False)

    # 把三个汇总字典变成三行表格，并以因子名作为索引。
    summary = pd.DataFrame(results).set_index("factor")
    summary.to_csv(OUTPUT_DIR / "factor_summary.csv")
    combined_figure = None
    # 未选择单因子原生报告时，生成三因子对比图。
    if not NATIVE_TEAR_SHEET_FACTOR:
        combined_figure = create_combined_chart(plot_results)

    print("\n三个因子的诊断汇总：")
    print(summary.round(6).to_string())
    print(f"\n结果已保存到：{OUTPUT_DIR}")
    print("确认：当前结果只使用用户提供或 Tushare 下载的真实前复权收盘价。")

    # VS Code 中选择“原生 tear sheet”配置时，会进入这个分支。
    if NATIVE_TEAR_SHEET_FACTOR:
        if NATIVE_TEAR_SHEET_FACTOR not in clean_factor_data:
            valid_names = ", ".join(clean_factor_data)
            raise ValueError(
                "原生 tear sheet 因子名称无效。可选值：" + valid_names
            )
        print(f"正在打开 {NATIVE_TEAR_SHEET_FACTOR} 的 Alphalens 原生报告。")
        # tears.py 会把收益、IC、换手率和自相关等分析组织成一份报告。
        al.tears.create_summary_tear_sheet(
            clean_factor_data[NATIVE_TEAR_SHEET_FACTOR],
            long_short=True,
            group_neutral=False,
        )
    elif SHOW_PLOTS:
        # plt.show() 会阻塞程序，直到用户把图形窗口关闭。
        print("图形窗口已经打开；关闭窗口后程序才会结束。")
        if AUTO_CLOSE_SECONDS > 0:
            timer = combined_figure.canvas.new_timer(
                interval=int(AUTO_CLOSE_SECONDS * 1000)
            )
            timer.add_callback(plt.close, combined_figure)
            timer.start()
        plt.show()
    else:
        plt.close(combined_figure)


if __name__ == "__main__":
    # 只有直接运行本文件时才执行 main()；被其他文件 import 时不会自动运行。
    # Alphalens 0.4.0 与旧版依赖会产生 FutureWarning，不影响本示例结果。
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
