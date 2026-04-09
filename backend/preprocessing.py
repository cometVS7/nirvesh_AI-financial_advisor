from __future__ import annotations

import numpy as np
import pandas as pd


class DataPreprocessor:
    def clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame.columns = [str(col).strip().lower().replace(" ", "_") for col in frame.columns]
        if "date" not in frame.columns:
            raise ValueError("Historical stock data is missing the 'date' column.")
        if "close" not in frame.columns:
            raise ValueError("Historical stock data is missing the 'close' column.")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        # Some Yahoo responses or cached files may only contain date/close.
        # Reconstruct minimal OHLCV so downstream feature engineering can proceed.
        if "open" not in frame.columns:
            frame["open"] = frame["close"]
        if "high" not in frame.columns:
            frame["high"] = frame[["open", "close"]].max(axis=1)
        if "low" not in frame.columns:
            frame["low"] = frame[["open", "close"]].min(axis=1)
        if "volume" not in frame.columns:
            frame["volume"] = 0
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
        frame["volume"] = frame["volume"].fillna(frame["volume"].median())
        return frame.drop_duplicates(subset=["date"]).reset_index(drop=True)

    def attach_context_scores(
        self,
        stock_df: pd.DataFrame,
        sentiment_score: float,
        sector_impact: float,
        regulatory_score: float,
        policy_score: float,
    ) -> pd.DataFrame:
        frame = stock_df.copy()
        frame["sentiment_score"] = float(sentiment_score)
        frame["sector_impact"] = float(sector_impact)
        frame["regulatory_score"] = float(regulatory_score)
        frame["policy_score"] = float(policy_score)
        frame["composite_context"] = (
            0.35 * frame["sentiment_score"]
            + 0.25 * frame["sector_impact"]
            + 0.2 * frame["policy_score"]
            + 0.2 * frame["regulatory_score"]
        )
        frame["context_trend"] = frame["composite_context"].rolling(5, min_periods=1).mean()
        frame["weekday"] = frame["date"].dt.weekday
        frame["month"] = frame["date"].dt.month
        frame["quarter"] = frame["date"].dt.quarter
        frame["price_range"] = frame["high"] - frame["low"]
        frame["price_gap"] = frame["open"] - frame["close"].shift(1).fillna(frame["open"])
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        return frame.bfill().ffill()
