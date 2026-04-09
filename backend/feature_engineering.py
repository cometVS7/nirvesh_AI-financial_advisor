from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "daily_return",
    "sma_10",
    "sma_20",
    "ema_12",
    "ema_26",
    "rsi",
    "macd",
    "macd_signal",
    "bollinger_high",
    "bollinger_low",
    "volatility_20",
    "price_range",
    "price_gap",
    "sentiment_score",
    "sector_impact",
    "regulatory_score",
    "policy_score",
    "composite_context",
    "context_trend",
    "weekday",
    "month",
    "quarter",
]


class FeatureEngineer:
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame["daily_return"] = frame["close"].pct_change().fillna(0.0)
        frame["sma_10"] = frame["close"].rolling(10).mean()
        frame["sma_20"] = frame["close"].rolling(20).mean()
        frame["ema_12"] = frame["close"].ewm(span=12, adjust=False).mean()
        frame["ema_26"] = frame["close"].ewm(span=26, adjust=False).mean()
        frame["macd"] = frame["ema_12"] - frame["ema_26"]
        frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
        frame["volatility_20"] = frame["daily_return"].rolling(20).std()
        rolling_mean = frame["close"].rolling(20).mean()
        rolling_std = frame["close"].rolling(20).std()
        frame["bollinger_high"] = rolling_mean + (2 * rolling_std)
        frame["bollinger_low"] = rolling_mean - (2 * rolling_std)
        delta = frame["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        frame["rsi"] = 100 - (100 / (1 + rs))
        frame["target_close"] = frame["close"].shift(-1)
        frame["target_return"] = frame["target_close"] / frame["close"] - 1
        frame["risk_target"] = pd.cut(frame["target_return"].fillna(0.0), bins=[-np.inf, -0.01, 0.02, np.inf], labels=["high", "mid", "low"])
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        frame = frame.dropna(subset=["target_close"]).reset_index(drop=True)
        frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].bfill().ffill()
        return frame
