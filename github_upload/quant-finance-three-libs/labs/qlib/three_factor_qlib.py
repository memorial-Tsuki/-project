"""使用 Qlib 和 LightGBM 学习三个真实股票因子。

本文件读取与 Alphalens、Backtrader 相同的 Tushare 沪深300前复权收盘价，
计算20日动量、5日反转和20日低波动三个因子，再用Qlib的数据集接口组织
训练集、验证集和测试集，最后用Qlib封装的LightGBM模型预测未来5日收益。

时间划分：
- 训练集：2021-02-01 至 2023-12-31；
- 验证集：2024-01-01 至 2024-12-31；
- 测试集：2025-01-01 至 2025-12-31。

本程序使用 StaticDataLoader，适合已有pandas表格的初学者。它仍然经过Qlib的
DataHandlerLP、DatasetH和LGBModel完整流程，只是不要求先制作Qlib二进制行情库。

输出文件位于 output/qlib_three_factors：
- qlib_metrics.csv：测试集Rank IC、ICIR和分组收益；
- qlib_predictions.csv：2025年每只股票的预测值与真实未来收益；
- qlib_daily_rank_ic.csv：每日Rank IC；
- qlib_portfolio_returns.csv：每5日一次的预测分组收益；
- qlib_feature_importance.csv：三个因子的模型重要性；
- qlib_three_factor_result.png：四联诊断图。
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
from typing import cast
import warnings

import matplotlib
import numpy as np
import pandas as pd
import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader
from qlib.workflow.recorder import MLflowRecorder

# 导入qLib模型模块时，它会列出未安装的CatBoost、XGBoost和PyTorch模型。
# 本例只使用已经安装的LightGBM，所以把这些与本任务无关的可选提示隐藏起来。
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    from qlib.contrib.model.gbdt import LGBModel


# ---------------------------------------------------------------------------
# 1. 配置
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 如果以后换成你提供的另一份真实数据，可以设置 QLIB_PRICE_CSV；
# 没有设置时继续使用当前 Tushare 沪深300文件。
PRICE_CSV_TEXT = os.environ.get("QLIB_PRICE_CSV", "").strip()
DEFAULT_PRICE_CSV = (
    PROJECT_ROOT
    / "data"
    / "tushare_csi300"
    / "csi300_qfq_close_2021_2025.csv"
)
PRICE_CSV = Path(PRICE_CSV_TEXT) if PRICE_CSV_TEXT else DEFAULT_PRICE_CSV
OUTPUT_DIR = PROJECT_ROOT / "output" / "qlib_three_factors"
# StaticDataLoader不读取Qlib二进制行情，但LGBModel仍会调用Qlib实验记录器。
# 因此准备一个存在的本地目录供qlib.init注册全局配置，避免指向用户目录。
QLIB_PROVIDER_PLACEHOLDER = PROJECT_ROOT / "data" / "qlib_static_provider"

# 把matplotlib缓存放在项目目录，避免用户主目录权限问题。
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

SHOW_PLOTS = os.environ.get("QLIB_SHOW_PLOTS", "1") != "0"
AUTO_CLOSE_SECONDS = float(os.environ.get("QLIB_AUTO_CLOSE_SECONDS", "0"))
matplotlib.use("Qt5Agg" if SHOW_PLOTS else "Agg")
import matplotlib.pyplot as plt


# Qlib DatasetH使用这些时间段切分同一个数据集。
SEGMENTS = {
    "train": ("2021-02-01", "2023-12-31"),
    "valid": ("2024-01-01", "2024-12-31"),
    "test": ("2025-01-01", "2025-12-31"),
}
LABEL_HORIZON = 5
PORTFOLIO_TOP_N = 20
PORTFOLIO_REBALANCE_DAYS = 5
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# 2. 读取真实行情与计算因子
# ---------------------------------------------------------------------------

def load_real_prices(csv_path: Path) -> pd.DataFrame:
    """读取真实价格；文件缺失时明确报错，不生成任何随机行情。"""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"未找到真实行情文件：{csv_path}\n"
            "请先运行 Alphalens 目录中的 download_tushare_csi300.py。"
        )

    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    prices.index.name = "datetime"
    prices.columns.name = "instrument"
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
    """计算三个原始因子，方向统一为数值越高越好。"""
    daily_returns = cast(
        pd.DataFrame,
        prices.pct_change(fill_method=None),
    )
    return {
        "momentum_20d": cast(
            pd.DataFrame,
            prices / prices.shift(20) - 1,
        ),
        "reversal_5d": cast(
            pd.DataFrame,
            -(prices / prices.shift(5) - 1),
        ),
        "low_volatility_20d": cast(
            pd.DataFrame,
            -daily_returns.rolling(20, min_periods=20).std(),
        ),
    }


def stack_wide_table(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """把“日期×股票”宽表转换为Qlib常用的双层索引长表。

    future_stack=True使用pandas的新stack实现，并保留缺失项，之后统一清洗。
    最终索引顺序是(datetime, instrument)。
    """
    # pandas的类型声明认为stack可能返回Series或DataFrame；本函数输入是普通
    # 单层列宽表，因此运行结果确定为Series，用cast把这一事实告诉Pylance。
    series = cast(pd.Series, frame.stack(future_stack=True))
    series.index.names = ["datetime", "instrument"]
    return series.to_frame(name=value_name)


def build_qlib_table(prices: pd.DataFrame) -> pd.DataFrame:
    """生成Qlib模型需要的三列特征和一列未来收益标签。"""
    raw_factors = calculate_three_factors(prices)

    # 每日横截面百分位排名消除了三个因子的量纲差异，范围约为0到1。
    # 排名仍然保持“数值越高越好”的方向。
    ranked_factors = {
        name: factor.rank(axis=1, method="average", pct=True)
        for name, factor in raw_factors.items()
    }
    feature_frames = [
        stack_wide_table(factor, name)
        for name, factor in ranked_factors.items()
    ]
    features = pd.concat(feature_frames, axis=1)

    # shift(-5)把5个交易日后的价格移动到今天这一行。
    # 标签只用于训练和评估，绝不能放进特征中。
    future_return_5d = prices.shift(-LABEL_HORIZON) / prices - 1.0
    label = stack_wide_table(future_return_5d, "future_return_5d")

    table = features.join(label, how="inner")
    # 为了让示例清晰，只有三个特征和标签都有效的行才进入模型。
    table = table.replace([np.inf, -np.inf], np.nan).dropna()
    table = table.sort_index()
    return table


# ---------------------------------------------------------------------------
# 3. 使用Qlib组织数据并训练模型
# ---------------------------------------------------------------------------

def create_qlib_dataset(table: pd.DataFrame) -> DatasetH:
    """把pandas长表装入StaticDataLoader、DataHandlerLP和DatasetH。

    StaticDataLoader的字典键会成为列的第一层：feature或label。
    Qlib模型因此知道哪些列可以作为输入，哪一列是预测目标。
    """
    feature_frame = table[
        ["momentum_20d", "reversal_5d", "low_volatility_20d"]
    ]
    label_frame = table[["future_return_5d"]]

    loader = StaticDataLoader(
        {
            "feature": feature_frame,
            "label": label_frame,
        }
    )
    handler = DataHandlerLP(
        data_loader=loader,
        infer_processors=[],
        learn_processors=[],
    )
    return DatasetH(handler=handler, segments=SEGMENTS)


def train_model(dataset: DatasetH) -> LGBModel:
    """训练Qlib封装的LightGBM回归模型。

    seed只控制模型训练的可复现性，不会生成或修改股票价格。
    deterministic和force_col_wise让相同数据、相同环境的结果尽量一致。
    """
    model = LGBModel(
        loss="mse",
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        lambda_l1=0.0,
        lambda_l2=1.0,
        seed=RANDOM_SEED,
        feature_fraction_seed=RANDOM_SEED,
        bagging_seed=RANDOM_SEED,
        data_random_seed=RANDOM_SEED,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        num_boost_round=300,
        early_stopping_rounds=30,
    )
    evaluation_history = {}
    model.fit(
        dataset,
        num_boost_round=300,
        early_stopping_rounds=30,
        verbose_eval=50,
        evals_result=evaluation_history,
    )
    return model


# ---------------------------------------------------------------------------
# 4. 测试集评估
# ---------------------------------------------------------------------------

def prepare_test_predictions(
    model: LGBModel, dataset: DatasetH
) -> pd.DataFrame:
    """对2025测试集预测，并与真实未来5日收益按索引对齐。"""
    raw_prediction = model.predict(dataset, segment="test")
    if isinstance(raw_prediction, pd.DataFrame):
        prediction = cast(
            pd.Series,
            raw_prediction[raw_prediction.columns[0]],
        ).copy()
    else:
        prediction = pd.Series(raw_prediction).copy()
    prediction.name = "prediction"

    # 本项目每次只请求一个test分段和label列组，Qlib运行时确定返回DataFrame。
    label_frame = cast(
        pd.DataFrame,
        dataset.prepare("test", col_set="label"),
    )
    label = cast(
        pd.Series,
        label_frame[label_frame.columns[0]],
    ).copy()
    label.name = "future_return_5d"
    result = pd.concat([prediction, label], axis=1).dropna().sort_index()
    return result


def calculate_daily_rank_ic(predictions: pd.DataFrame) -> pd.Series:
    """逐日计算预测排名与未来收益排名的Spearman相关系数。"""
    daily_ic = cast(
        pd.Series,
        predictions.groupby(level="datetime").apply(
            lambda frame: frame["prediction"].corr(
                frame["future_return_5d"], method="spearman"
            )
        ),
    )
    daily_ic.name = "rank_ic"
    return daily_ic.dropna()


def calculate_non_overlapping_portfolios(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """每隔5个交易日构造一次预测最高组和最低组。

    标签本身是未来5日收益，因此每5日取一个截面，避免把大量重叠持有期直接
    连乘。long_short是研究诊断组合；A股普通账户通常不能直接做空股票。
    """
    records = []
    grouped = predictions.groupby(level="datetime")
    for number, (date, frame) in enumerate(grouped):
        if number % PORTFOLIO_REBALANCE_DAYS != 0:
            continue
        frame = frame.sort_values("prediction")
        group_size = min(PORTFOLIO_TOP_N, max(1, len(frame) // 5))
        bottom_return = frame.head(group_size)["future_return_5d"].mean()
        top_return = frame.tail(group_size)["future_return_5d"].mean()
        records.append(
            {
                "date": pd.Timestamp(str(date)),
                "top_return_5d": top_return,
                "bottom_return_5d": bottom_return,
                "long_short_return_5d": top_return - bottom_return,
                "stocks_per_side": group_size,
            }
        )
    return pd.DataFrame(records).set_index("date")


def calculate_prediction_quantiles(
    predictions: pd.DataFrame,
) -> pd.Series:
    """每天按预测值分成5组，计算各组平均未来收益。"""
    working = predictions.copy()

    def assign_quantile(series: pd.Series) -> pd.Series:
        # rank(method='first')避免大量相同预测值导致qcut边界重复。
        ranks = series.rank(method="first")
        return cast(
            pd.Series,
            pd.qcut(ranks, q=5, labels=[1, 2, 3, 4, 5]),
        )

    working["prediction_quantile"] = working.groupby(
        level="datetime"
    )["prediction"].transform(assign_quantile)
    return cast(
        pd.Series,
        working.groupby("prediction_quantile", observed=True)[
            "future_return_5d"
        ].mean(),
    )


def save_evaluation(
    model: LGBModel,
    predictions: pd.DataFrame,
    daily_ic: pd.Series,
    portfolio_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """保存测试集预测、指标和特征重要性。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_importance = cast(
        pd.Series,
        model.get_feature_importance(importance_type="gain"),
    ).copy()
    # 直接设置Series.name，避免pandas的rename重载被Pylance误判为“重命名索引”。
    feature_importance.name = "gain_importance"
    # StaticDataLoader进入LightGBM后可能把列名转成Column_0等通用名称。
    # 根据固定的特征顺序恢复成适合阅读的因子名。
    generic_name_map = {
        "Column_0": "momentum_20d",
        "Column_1": "reversal_5d",
        "Column_2": "low_volatility_20d",
    }
    feature_importance = feature_importance.rename(index=generic_name_map)
    prediction_quantiles = calculate_prediction_quantiles(predictions)

    ic_std = daily_ic.std(ddof=1)
    metrics = pd.DataFrame(
        [
            {
                "test_start": SEGMENTS["test"][0],
                "test_end": SEGMENTS["test"][1],
                "test_observations": len(predictions),
                "daily_rank_ic_mean": daily_ic.mean(),
                "daily_rank_ic_std": ic_std,
                "daily_icir": daily_ic.mean() / ic_std
                if ic_std > 0
                else np.nan,
                "rank_ic_positive_ratio": (daily_ic > 0).mean(),
                "top_group_mean_5d": portfolio_returns[
                    "top_return_5d"
                ].mean(),
                "bottom_group_mean_5d": portfolio_returns[
                    "bottom_return_5d"
                ].mean(),
                "long_short_mean_5d": portfolio_returns[
                    "long_short_return_5d"
                ].mean(),
            }
        ]
    )

    predictions.reset_index().to_csv(
        OUTPUT_DIR / "qlib_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily_ic.to_csv(
        OUTPUT_DIR / "qlib_daily_rank_ic.csv", encoding="utf-8-sig"
    )
    portfolio_returns.to_csv(
        OUTPUT_DIR / "qlib_portfolio_returns.csv", encoding="utf-8-sig"
    )
    feature_importance.to_csv(
        OUTPUT_DIR / "qlib_feature_importance.csv", encoding="utf-8-sig"
    )
    prediction_quantiles_for_csv = prediction_quantiles.copy()
    prediction_quantiles_for_csv.name = "mean_future_return_5d"
    prediction_quantiles_for_csv.to_csv(
        OUTPUT_DIR / "qlib_prediction_quantile_returns.csv",
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        OUTPUT_DIR / "qlib_metrics.csv", index=False, encoding="utf-8-sig"
    )
    return metrics, feature_importance, prediction_quantiles


# ---------------------------------------------------------------------------
# 5. 绘图
# ---------------------------------------------------------------------------

def plot_evaluation(
    daily_ic: pd.Series,
    portfolio_returns: pd.DataFrame,
    feature_importance: pd.Series,
    prediction_quantiles: pd.Series,
) -> None:
    """绘制Qlib模型的四个核心诊断图。"""
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))

    feature_importance.sort_values().plot(
        kind="barh", ax=axes[0, 0], color="#4c72b0"
    )
    axes[0, 0].set_title("Qlib LightGBM feature importance")
    axes[0, 0].set_xlabel("Gain importance")

    daily_ic.plot(ax=axes[0, 1], alpha=0.30, color="#777777")
    rolling_ic = cast(
        pd.Series,
        daily_ic.rolling(20, min_periods=5).mean(),
    )
    rolling_ic.plot(
        ax=axes[0, 1], color="#c44e52", linewidth=1.8
    )
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Daily Rank IC and 20-day mean")
    axes[0, 1].set_ylabel("Rank IC")

    cumulative = (1.0 + portfolio_returns[
        ["top_return_5d", "bottom_return_5d", "long_short_return_5d"]
    ]).cumprod()
    cumulative.plot(ax=axes[1, 0], linewidth=1.4)
    axes[1, 0].set_title("Non-overlapping 5-day diagnostic portfolios")
    axes[1, 0].set_ylabel("Growth of 1 unit")
    axes[1, 0].set_xlabel("Date")

    (prediction_quantiles * 100.0).plot(
        kind="bar", ax=axes[1, 1], color="#55a868"
    )
    axes[1, 1].set_title("Mean future return by prediction quantile")
    axes[1, 1].set_xlabel("Prediction quantile (1=lowest, 5=highest)")
    axes[1, 1].set_ylabel("Mean future 5-day return (%)")
    axes[1, 1].tick_params(axis="x", rotation=0)

    for axis in axes.flat:
        axis.grid(alpha=0.20)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "qlib_three_factor_result.png", dpi=160)

    if SHOW_PLOTS:
        print("Qlib图形窗口已经打开；关闭窗口后程序才会结束。")
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
    """执行完整的Qlib三因子机器学习流程。"""
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QLIB_PROVIDER_PLACEHOLDER.mkdir(parents=True, exist_ok=True)

    # 初始化Qlib的全局配置和实验记录器。后续DatasetH的数据实际来自下面的
    # StaticDataLoader，而不是这个占位目录。
    # 文件式MLflow存储对本地教学最简单。过滤它在2026年新增的迁移提醒，
    # 不影响当前实验结果；所有实验记录统一放入项目output目录。
    warnings.filterwarnings(
        "ignore",
        message="The filesystem tracking backend.*",
        category=FutureWarning,
    )
    # 官方GitHub源码是以普通文件夹下载的，没有.git目录。Qlib默认会执行
    # git diff保存“未提交代码快照”，这会打印与模型无关的大段Git帮助信息。
    # 仅关闭这一项可选快照；MLflow中的模型指标记录仍然保留。
    MLflowRecorder._log_uncommitted_code = lambda self: None
    experiment_uri = f"file:{OUTPUT_DIR / 'mlruns'}"
    qlib.init(
        provider_uri=str(QLIB_PROVIDER_PLACEHOLDER),
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": experiment_uri,
                "default_exp_name": "three_factor_learning",
            },
        },
    )

    prices = load_real_prices(PRICE_CSV)
    table = build_qlib_table(prices)
    print(f"可用于Qlib的数据行数：{len(table):,}")
    print(f"Qlib版本：{qlib.__version__}")

    dataset = create_qlib_dataset(table)
    model = train_model(dataset)
    predictions = prepare_test_predictions(model, dataset)
    daily_ic = calculate_daily_rank_ic(predictions)
    portfolio_returns = calculate_non_overlapping_portfolios(predictions)
    metrics, feature_importance, prediction_quantiles = save_evaluation(
        model,
        predictions,
        daily_ic,
        portfolio_returns,
    )

    print("\nQlib测试集核心指标：")
    print(metrics.round(6).to_string(index=False))
    print("\n特征重要性：")
    print(feature_importance.round(2).to_string())
    print(f"\n结果已保存到：{OUTPUT_DIR}")
    print("确认：本实验使用Tushare真实沪深300前复权收盘价，不使用随机行情。")

    plot_evaluation(
        daily_ic,
        portfolio_returns,
        feature_importance,
        prediction_quantiles,
    )


if __name__ == "__main__":
    main()
