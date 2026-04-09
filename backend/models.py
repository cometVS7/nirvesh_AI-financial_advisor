from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from backend.config import settings
from backend.feature_engineering import FEATURE_COLUMNS

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential, load_model
except ImportError:  # pragma: no cover
    Sequential = None
    load_model = None


@dataclass
class ModelArtifacts:
    regressor: object
    random_forest: RandomForestClassifier
    scaler: StandardScaler
    lstm_scaler: MinMaxScaler
    lstm_model_path: Path | None
    metrics: Dict[str, float]
    uses_tensorflow: bool


class MarketModelPipeline:
    def __init__(self) -> None:
        self.model_root = settings.models_dir
        self.model_root.mkdir(parents=True, exist_ok=True)

    def train_or_load(self, ticker: str, frame: pd.DataFrame, context_signature: Dict | None = None) -> ModelArtifacts:
        ticker_dir = self.model_root / ticker.replace(".", "_")
        ticker_dir.mkdir(parents=True, exist_ok=True)
        regressor_path = ticker_dir / "xgb.joblib"
        rf_path = ticker_dir / "rf.joblib"
        scaler_path = ticker_dir / "scaler.joblib"
        lstm_scaler_path = ticker_dir / "lstm_scaler.joblib"
        lstm_model_path = ticker_dir / "lstm.keras"
        metrics_path = ticker_dir / "metrics.json"
        metadata_path = ticker_dir / "metadata.json"

        if (
            regressor_path.exists()
            and rf_path.exists()
            and scaler_path.exists()
            and lstm_scaler_path.exists()
            and self._model_is_current(metadata_path, frame, context_signature)
        ):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            return ModelArtifacts(
                regressor=joblib.load(str(regressor_path)),
                random_forest=joblib.load(str(rf_path)),
                scaler=joblib.load(str(scaler_path)),
                lstm_scaler=joblib.load(str(lstm_scaler_path)),
                lstm_model_path=lstm_model_path if lstm_model_path.exists() else None,
                metrics=metrics,
                uses_tensorflow=bool(lstm_model_path.exists() and load_model is not None),
            )

        artifacts = self._train_models(frame, lstm_model_path)
        joblib.dump(artifacts.regressor, str(regressor_path))
        joblib.dump(artifacts.random_forest, str(rf_path))
        joblib.dump(artifacts.scaler, str(scaler_path))
        joblib.dump(artifacts.lstm_scaler, str(lstm_scaler_path))
        metrics_path.write_text(json.dumps(artifacts.metrics, indent=2), encoding="utf-8")
        metadata_path.write_text(json.dumps(self._training_metadata(frame, context_signature), indent=2), encoding="utf-8")
        return artifacts

    def predict(
        self,
        ticker: str,
        frame: pd.DataFrame,
        horizon_days: int = 30,
        live_price: float | None = None,
        quote_meta: Dict | None = None,
        context_score: float = 0.0,
        context_signature: Dict | None = None,
    ) -> Dict:
        artifacts = self.train_or_load(ticker, frame, context_signature=context_signature)
        return self._predict_with_artifacts(
            artifacts=artifacts,
            frame=frame,
            horizon_days=horizon_days,
            live_price=live_price,
            quote_meta=quote_meta,
            context_score=context_score,
        )

    def predict_multi_horizon(
        self,
        ticker: str,
        frame: pd.DataFrame,
        horizons: list[int],
        live_price: float | None = None,
        quote_meta: Dict | None = None,
        context_score: float = 0.0,
        context_signature: Dict | None = None,
    ) -> Dict[int, Dict]:
        artifacts = self.train_or_load(ticker, frame, context_signature=context_signature)
        results: Dict[int, Dict] = {}
        for horizon_days in sorted({int(day) for day in horizons if int(day) > 0}):
            results[horizon_days] = self._predict_with_artifacts(
                artifacts=artifacts,
                frame=frame,
                horizon_days=horizon_days,
                live_price=live_price,
                quote_meta=quote_meta,
                context_score=context_score,
            )
        return results

    def predict_backtest(
        self,
        frame: pd.DataFrame,
        horizon_days: int = 1,
        context_score: float = 0.0,
    ) -> Dict:
        temp_lstm_path = self.model_root / "_backtest_temp.keras"
        artifacts = self._train_models(frame, temp_lstm_path)
        return self._predict_with_artifacts(
            artifacts=artifacts,
            frame=frame,
            horizon_days=horizon_days,
            live_price=None,
            quote_meta={},
            context_score=context_score,
        )

    def _predict_with_artifacts(
        self,
        artifacts: ModelArtifacts,
        frame: pd.DataFrame,
        horizon_days: int,
        live_price: float | None,
        quote_meta: Dict | None,
        context_score: float,
    ) -> Dict:
        latest_features = frame[FEATURE_COLUMNS].iloc[-1:]
        scaled_latest = artifacts.scaler.transform(latest_features)
        xgb_price = float(artifacts.regressor.predict(scaled_latest)[0])
        rf_proba = artifacts.random_forest.predict_proba(scaled_latest)[0]
        rf_classes = list(artifacts.random_forest.classes_)
        risk_index = int(np.argmax(rf_proba))
        model_risk_label = str(rf_classes[risk_index])
        historical_close = float(frame["close"].iloc[-1])
        current_price = float(live_price) if live_price is not None else historical_close
        rf_price = historical_close * (1 + {"low": 0.07, "mid": 0.03, "high": -0.02}.get(model_risk_label, 0.02))
        lstm_price = self._predict_lstm(frame, artifacts)
        historical_final_price = 0.4 * lstm_price + 0.35 * xgb_price + 0.25 * rf_price
        projected_ratio = historical_final_price / historical_close if historical_close else 1.0
        context_multiplier = float(np.clip(1 + (context_score * 0.12), 0.82, 1.18))
        base_expected_return = (projected_ratio - 1.0) * context_multiplier
        horizon_scale = float(np.sqrt(max(horizon_days, 1) / 30.0))
        horizon_expected_return = float(np.clip(base_expected_return * horizon_scale, -0.45, 0.75))
        final_price = current_price * (1 + horizon_expected_return)
        growth_probability = float(np.clip((final_price / current_price - 1) * 3 + 0.5, 0.0, 1.0))
        realized_volatility = float(frame["daily_return"].tail(30).std()) if "daily_return" in frame.columns else 0.0
        recent_drawdown = float((frame["close"].tail(90) / frame["close"].tail(90).cummax() - 1).min()) if len(frame) >= 30 else 0.0
        risk_label = self._calibrate_risk_label(
            model_risk_label=model_risk_label,
            growth_probability=growth_probability,
            expected_return=horizon_expected_return if current_price else 0.0,
            realized_volatility=realized_volatility,
            recent_drawdown=recent_drawdown,
            context_score=context_score,
        )
        last_date = pd.to_datetime(frame["date"].iloc[-1])
        future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=horizon_days)
        predicted_series = self._build_prediction_series(
            frame=frame,
            current_price=current_price,
            final_price=float(final_price),
            horizon_days=horizon_days,
        )
        anchor_date = quote_meta.get("quote_time") if quote_meta else None
        if anchor_date:
            anchor_ts = pd.to_datetime(anchor_date, errors="coerce")
        else:
            anchor_ts = last_date
        anchor_point = {"date": pd.Timestamp(anchor_ts if pd.notna(anchor_ts) else last_date).strftime("%Y-%m-%d"), "predicted_close": round(float(current_price), 2)}
        return {
            "current_price": round(current_price, 2),
            "historical_reference_price": round(historical_close, 2),
            "predicted_price": round(float(final_price), 2),
            "xgboost_price": round(xgb_price, 2),
            "lstm_price": round(float(lstm_price), 2),
            "random_forest_price": round(float(rf_price), 2),
            "growth_probability": round(growth_probability, 4),
            "predicted_growth_pct": round((final_price / current_price - 1) * 100, 2),
            "risk_level": risk_label,
            "model_risk_level": model_risk_label,
            "risk_confidence": round(float(rf_proba[risk_index]), 4),
            "realized_volatility_30d": round(realized_volatility, 4),
            "recent_drawdown_90d": round(recent_drawdown, 4),
            "metrics": artifacts.metrics,
            "quote": quote_meta or {},
            "context_score": round(context_score, 4),
            "context_multiplier": round(context_multiplier, 4),
            "prediction_horizon_days": int(horizon_days),
            "prediction_series": [anchor_point] + [
                {"date": date.strftime("%Y-%m-%d"), "predicted_close": round(float(value), 2)}
                for date, value in zip(future_dates, predicted_series)
            ],
        }

    def _build_prediction_series(
        self,
        frame: pd.DataFrame,
        current_price: float,
        final_price: float,
        horizon_days: int,
    ) -> np.ndarray:
        if horizon_days <= 0:
            return np.array([], dtype=float)
        if horizon_days == 1:
            return np.array([final_price], dtype=float)

        returns = (
            frame["daily_return"].dropna()
            if "daily_return" in frame.columns
            else frame["close"].pct_change().dropna()
        )
        recent_returns = returns.tail(60)
        recent_vol = float(recent_returns.std()) if len(recent_returns) else 0.012
        recent_drift = float(recent_returns.mean()) if len(recent_returns) else 0.0005
        recent_vol = float(np.clip(recent_vol, 0.004, 0.04))
        recent_drift = float(np.clip(recent_drift, -0.015, 0.015))

        progress = np.linspace(0.0, 1.0, horizon_days)
        base_path = current_price + (final_price - current_price) * progress

        # Add market-like curvature using the stock's recent volatility profile.
        wave_1 = np.sin(np.linspace(0, 2.5 * np.pi, horizon_days))
        wave_2 = np.sin(np.linspace(0, 5.0 * np.pi, horizon_days) + np.pi / 6)
        volatility_shape = (0.65 * wave_1 + 0.35 * wave_2) * recent_vol * current_price * 1.35

        # Drift component keeps the path aligned with the stock's recent local trend.
        drift_shape = np.cumsum(np.full(horizon_days, recent_drift * current_price * 0.18))

        # Fade the wiggle to zero at both ends so the line starts at the live/current price
        # and lands exactly on the model target.
        taper = np.sin(np.pi * progress) ** 1.15
        series = base_path + (volatility_shape + drift_shape) * taper

        series[0] = current_price
        series[-1] = final_price

        # Smooth out any remaining harsh day-to-day jumps while preserving curvature.
        smoothed = pd.Series(series).rolling(window=3, min_periods=1, center=True).mean().to_numpy()
        smoothed[0] = current_price
        smoothed[-1] = final_price

        # Prevent unrealistic single-day cliffs.
        max_step = max(current_price * max(recent_vol * 1.8, 0.01), 0.5)
        for idx in range(1, len(smoothed)):
            delta = smoothed[idx] - smoothed[idx - 1]
            if delta > max_step:
                smoothed[idx] = smoothed[idx - 1] + max_step
            elif delta < -max_step:
                smoothed[idx] = smoothed[idx - 1] - max_step

        # Re-anchor exactly after clipping.
        adjustment = np.linspace(0.0, final_price - smoothed[-1], horizon_days)
        smoothed = smoothed + adjustment
        smoothed[0] = current_price
        smoothed[-1] = final_price
        return smoothed

    def _calibrate_risk_label(
        self,
        model_risk_label: str,
        growth_probability: float,
        expected_return: float,
        realized_volatility: float,
        recent_drawdown: float,
        context_score: float,
    ) -> str:
        risk_score = 0.0
        risk_score += {"low": -0.2, "mid": 0.1, "high": 0.35}.get(model_risk_label, 0.1)
        risk_score += min(realized_volatility / 0.03, 1.2) * 0.45
        risk_score += min(abs(recent_drawdown) / 0.2, 1.2) * 0.3
        if expected_return > 0.08 and growth_probability > 0.65:
            risk_score -= 0.18
        if context_score > 0.2:
            risk_score -= 0.08
        if expected_return < 0:
            risk_score += 0.15
        if risk_score <= 0.18:
            return "low"
        if risk_score <= 0.52:
            return "mid"
        return "high"

    def _training_metadata(self, frame: pd.DataFrame, context_signature: Dict | None = None) -> Dict[str, str | int | Dict]:
        latest_date = pd.to_datetime(frame["date"].iloc[-1]).strftime("%Y-%m-%d") if not frame.empty else ""
        return {
            "rows": int(len(frame)),
            "latest_date": latest_date,
            "context_signature": context_signature or {},
        }

    def _model_is_current(self, metadata_path: Path, frame: pd.DataFrame, context_signature: Dict | None = None) -> bool:
        if not metadata_path.exists():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        current = self._training_metadata(frame, context_signature)
        return (
            metadata.get("rows") == current["rows"]
            and metadata.get("latest_date") == current["latest_date"]
            and metadata.get("context_signature", {}) == current["context_signature"]
        )

    def _train_models(self, frame: pd.DataFrame, lstm_model_path: Path) -> ModelArtifacts:
        features = frame[FEATURE_COLUMNS]
        targets = frame["target_close"]
        risk_labels = frame["risk_target"].astype(str)
        split_idx = int(len(frame) * 0.8)
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]
        risk_train = risk_labels.iloc[:split_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        regressor = (
            XGBRegressor(
                n_estimators=80,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            )
            if XGBRegressor is not None
            else GradientBoostingRegressor(random_state=42)
        )
        regressor.fit(X_train_scaled, y_train)

        random_forest = RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=1)
        random_forest.fit(X_train_scaled, risk_train)

        xgb_test_pred = regressor.predict(X_test_scaled) if len(X_test_scaled) else regressor.predict(X_train_scaled[:1])
        xgb_mae = mean_absolute_error(y_test, xgb_test_pred) if len(y_test) else mean_absolute_error(y_train[:1], xgb_test_pred[:1])
        lstm_scaler = MinMaxScaler()
        lstm_scaler.fit(frame[["close"]])
        uses_tensorflow = bool(Sequential is not None)
        if uses_tensorflow:
            self._train_lstm(frame, lstm_scaler, lstm_model_path)
        else:
            lstm_model_path = None
        return ModelArtifacts(
            regressor=regressor,
            random_forest=random_forest,
            scaler=scaler,
            lstm_scaler=lstm_scaler,
            lstm_model_path=lstm_model_path if uses_tensorflow else None,
            metrics={"xgboost_mae": round(float(xgb_mae), 4)},
            uses_tensorflow=uses_tensorflow,
        )

    def _train_lstm(self, frame: pd.DataFrame, scaler: MinMaxScaler, output_path: Path) -> None:
        if Sequential is None:
            return
        scaled = scaler.transform(frame[["close"]])
        seq_len = 30
        X_seq, y_seq = [], []
        for idx in range(seq_len, len(scaled)):
            X_seq.append(scaled[idx - seq_len : idx])
            y_seq.append(scaled[idx])
        if len(X_seq) < 60:
            return
        X_array, y_array = np.array(X_seq), np.array(y_seq)
        split_idx = int(len(X_array) * 0.8)
        X_train, y_train = X_array[:split_idx], y_array[:split_idx]
        model = Sequential(
            [
                LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                LSTM(32),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)])
        model.save(str(output_path))

    def _predict_lstm(self, frame: pd.DataFrame, artifacts: ModelArtifacts) -> float:
        seq_len = 30
        scaled = artifacts.lstm_scaler.transform(frame[["close"]])
        if len(scaled) < seq_len + 1:
            return float(frame["close"].iloc[-1])
        if artifacts.uses_tensorflow and artifacts.lstm_model_path and load_model is not None and artifacts.lstm_model_path.exists():
            model = load_model(str(artifacts.lstm_model_path))
            pred = model.predict(scaled[-seq_len:].reshape(1, seq_len, 1), verbose=0)[0][0]
            return float(artifacts.lstm_scaler.inverse_transform(np.array([[pred]]))[0][0])
        last_close = float(frame["close"].iloc[-1])
        recent_trend = float(frame["close"].pct_change().tail(10).mean())
        return last_close * (1 + recent_trend)
