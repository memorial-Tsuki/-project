"""按 Tushare 官方接口下载真实沪深300前复权日线数据。

官方依据：
1. A股日线 daily：
   https://tushare.pro/document/2?doc_id=27
2. 通用行情 pro_bar（支持 adj='qfq' 前复权）：
   https://tushare.pro/document/2?doc_id=109
3. 指数成分和权重 index_weight：
   https://tushare.pro/document/2?doc_id=96

本脚本使用一个固定的沪深300成分股截面：
- 成分来源：2025 年 12 月 index_weight 返回的最新 trade_date；
- 行情区间：2021-01-01 至 2025-12-31；
- 价格口径：前复权日线收盘价（qfq close）。

这样做便于初学者复现，但会有“使用期末成分股”的幸存者偏差。脚本会把
这个限制写进 metadata.json，写报告时必须披露。后续可以升级成按月动态成分股。
"""

from datetime import datetime
import json
import os
from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm
import tushare as ts


# ---------------------------------------------------------------------------
# 1. 下载配置
# ---------------------------------------------------------------------------

# 这里使用 Tushare index_weight 官方示例中的沪深300代码。
INDEX_CODE = "399300.SZ"

# 用 2025 年 12 月最后一次公布的指数权重确定固定股票池。
WEIGHT_START_DATE = "20251201"
WEIGHT_END_DATE = "20251231"

# 五个完整自然年的研究区间。
PRICE_START_DATE = "20210101"
PRICE_END_DATE = "20251231"

# 默认下载全部约 300 个成分股。若只是测试，可临时设置环境变量
# TUSHARE_MAX_STOCKS=10；程序会按指数权重从高到低选择，不会随机抽取。
MAX_STOCKS = int(os.environ.get("TUSHARE_MAX_STOCKS", "300"))

# 每次请求之间暂停，降低触发接口频率限制的概率。
REQUEST_INTERVAL_SECONDS = 0.25

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "tushare_csi300"
RAW_DIR = DATA_DIR / "qfq_by_stock"
PRICE_OUTPUT = DATA_DIR / "csi300_qfq_close_2021_2025.csv"
QUALITY_OUTPUT = DATA_DIR / "data_quality.csv"
WEIGHT_OUTPUT = DATA_DIR / "csi300_constituent_weights_202512.csv"
METADATA_OUTPUT = DATA_DIR / "metadata.json"
ENV_FILE = Path(__file__).with_name(".env")


def read_token():
    """从环境变量或本地 .env 读取 Token，但绝不打印 Token。"""
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token

    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == "TUSHARE_TOKEN":
                token = value.strip().strip('"').strip("'")
                if token:
                    return token

    raise RuntimeError(
        "没有找到 TUSHARE_TOKEN。请打开 labs/alphalens/.env，"
        "只在 TUSHARE_TOKEN= 后粘贴你的 Token。"
    )


def get_constituents(pro):
    """调用官方 index_weight 接口取得指定月份的最新沪深300成分。"""
    weights = pro.index_weight(
        index_code=INDEX_CODE,
        start_date=WEIGHT_START_DATE,
        end_date=WEIGHT_END_DATE,
    )
    if weights is None or weights.empty:
        raise RuntimeError(
            "index_weight 没有返回数据。请检查积分权限、指数代码和日期。"
        )

    required = {"index_code", "con_code", "trade_date", "weight"}
    missing = required.difference(weights.columns)
    if missing:
        raise RuntimeError(f"index_weight 缺少官方字段：{sorted(missing)}")

    latest_trade_date = str(weights["trade_date"].max())
    latest = (
        weights.loc[weights["trade_date"].astype(str) == latest_trade_date]
        .drop_duplicates(subset="con_code")
        .sort_values("weight", ascending=False)
        .head(MAX_STOCKS)
        .reset_index(drop=True)
    )
    latest.to_csv(WEIGHT_OUTPUT, index=False, encoding="utf-8-sig")
    print(
        f"指数成分日期：{latest_trade_date}；"
        f"本次下载股票数：{len(latest)}"
    )
    return latest, latest_trade_date


def download_one_stock(ts_code, pro):
    """按官方 pro_bar 示例下载一只股票的前复权日线。"""
    cache_file = RAW_DIR / f"{ts_code.replace('.', '_')}.csv"
    if cache_file.exists():
        cached = pd.read_csv(cache_file, dtype={"trade_date": str})
        if not cached.empty:
            return cached

    # 官方参数：asset='E' 表示股票，freq='D' 表示日线，adj='qfq' 表示前复权。
    frame = ts.pro_bar(
        ts_code=ts_code,
        api=pro,
        start_date=PRICE_START_DATE,
        end_date=PRICE_END_DATE,
        asset="E",
        freq="D",
        adj="qfq",
        retry_count=3,
    )
    if frame is None or frame.empty:
        return pd.DataFrame()

    required = {"ts_code", "trade_date", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{ts_code} 的 pro_bar 结果缺少字段：{sorted(missing)}")

    frame = frame.sort_values("trade_date").drop_duplicates("trade_date")
    frame.to_csv(cache_file, index=False, encoding="utf-8-sig")
    return frame


def build_price_matrix(members, pro):
    """逐只下载并拼成 Alphalens 需要的“日期×股票”价格宽表。"""
    close_series = []
    quality_rows = []

    for ts_code in tqdm(members["con_code"], desc="下载前复权日线"):
        try:
            frame = download_one_stock(ts_code, pro)
            if frame.empty:
                quality_rows.append(
                    {"ts_code": ts_code, "rows": 0, "status": "empty"}
                )
                continue

            dates = pd.to_datetime(frame["trade_date"], format="%Y%m%d")
            close = pd.Series(
                pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
                index=dates,
                name=ts_code,
            ).dropna()
            close = close[~close.index.duplicated(keep="last")].sort_index()
            close_series.append(close)
            quality_rows.append(
                {
                    "ts_code": ts_code,
                    "rows": len(close),
                    "first_date": close.index.min().date().isoformat(),
                    "last_date": close.index.max().date().isoformat(),
                    "status": "ok",
                }
            )
        except Exception as error:
            quality_rows.append(
                {
                    "ts_code": ts_code,
                    "rows": 0,
                    "status": f"error: {type(error).__name__}: {error}",
                }
            )
        time.sleep(REQUEST_INTERVAL_SECONDS)

    quality = pd.DataFrame(quality_rows)
    quality.to_csv(QUALITY_OUTPUT, index=False, encoding="utf-8-sig")

    if not close_series:
        raise RuntimeError("所有股票行情都下载失败，请查看 data_quality.csv。")

    prices = pd.concat(close_series, axis=1).sort_index()
    prices.index.name = "date"
    prices = prices.loc[PRICE_START_DATE:PRICE_END_DATE]
    prices.to_csv(PRICE_OUTPUT, encoding="utf-8-sig")
    return prices, quality


def save_metadata(prices, quality, constituent_date):
    """保存数据来源、参数和局限，保证课程报告可以复现。"""
    metadata = {
        "source": "Tushare Pro",
        "official_daily_document": "https://tushare.pro/document/2?doc_id=27",
        "official_pro_bar_document": "https://tushare.pro/document/2?doc_id=109",
        "official_index_weight_document": "https://tushare.pro/document/2?doc_id=96",
        "sdk_version": ts.__version__,
        "index_code": INDEX_CODE,
        "constituent_trade_date": constituent_date,
        "price_start_date": PRICE_START_DATE,
        "price_end_date": PRICE_END_DATE,
        "frequency": "D",
        "adjustment": "qfq",
        "price_field": "close",
        "rows": int(prices.shape[0]),
        "columns": int(prices.shape[1]),
        "missing_ratio": float(prices.isna().sum().sum() / prices.size),
        "successful_assets": int((quality["status"] == "ok").sum()),
        "created_at": datetime.now().astimezone().isoformat(),
        "known_limitation": (
            "使用2025年12月最新一期沪深300成分作为2021-2025固定股票池，"
            "存在期末成分选择导致的幸存者偏差；后续可升级为按月动态成分。"
        ),
    }
    METADATA_OUTPUT.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    """连接接口、下载、做质量检查并保存可复现的数据快照。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    token = read_token()
    ts.set_token(token)
    pro = ts.pro_api(token)

    members, constituent_date = get_constituents(pro)
    prices, quality = build_price_matrix(members, pro)
    save_metadata(prices, quality, constituent_date)

    print("\n真实行情下载完成：")
    print(f"价格文件：{PRICE_OUTPUT}")
    print(f"数据形状：{prices.shape[0]} 个交易日 × {prices.shape[1]} 只股票")
    print(f"总体缺失率：{prices.isna().sum().sum() / prices.size:.2%}")
    print(f"质量报告：{QUALITY_OUTPUT}")
    print(f"元数据：{METADATA_OUTPUT}")


if __name__ == "__main__":
    main()
