from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List

import pandas as pd

from backend.agents import FinancialAgentOrchestrator
from backend.config import HOLDING_WINDOWS, RISK_MAP, SECTORS, SECTOR_LEADERS, ensure_directories, settings
from backend.data_collection import DataCollector
from backend.feature_engineering import FeatureEngineer
from backend.models import MarketModelPipeline
from backend.nlp import FinBertSentimentAnalyzer
from backend.preprocessing import DataPreprocessor

CACHE: Dict[str, Dict[str, object]] = {}
CACHE_TTL_SECONDS = 600
REPEATED_REQUEST_TTL_SECONDS = 300
LAST_COMPANY_REQUEST = {
    "company": None,
    "budget": None,
    "timestamp": 0.0,
    "result": None,
}


def is_cache_valid(timestamp: float, ttl: int) -> bool:
    return bool(timestamp) and (time.time() - timestamp) <= ttl


def get_cache(key: str, ttl: int = CACHE_TTL_SECONDS):
    entry = CACHE.get(key)
    if not entry:
        return None
    if not is_cache_valid(float(entry.get("timestamp", 0.0)), ttl):
        CACHE.pop(key, None)
        return None
    return copy.deepcopy(entry.get("data"))


def set_cache(key: str, data) -> None:
    CACHE[key] = {"data": copy.deepcopy(data), "timestamp": time.time()}


class FinancialAdvisorService:
    def __init__(self) -> None:
        ensure_directories()
        self.collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_pipeline = MarketModelPipeline()
        self.sentiment_engine = FinBertSentimentAnalyzer()
        self.agent_system = FinancialAgentOrchestrator()

    def list_sectors(self) -> List[str]:
        return SECTORS

    def market_mood(self) -> Dict:
        return self.collector.fetch_market_mood_snapshot()

    def portfolio_recommendation(
        self,
        budget: float,
        risk_tolerance: str,
        holding_period: str,
        selected_sectors: List[str] | None = None,
    ) -> Dict:
        if budget <= 0:
            raise ValueError("Budget must be greater than zero.")
        normalized_risk = risk_tolerance.strip().lower()
        if normalized_risk not in {"low", "mid", "high"}:
            normalized_risk = "mid"
        horizon_days = self._holding_period_to_horizon(holding_period)
        normalized_sectors = self._normalize_selected_sectors(selected_sectors)
        if not normalized_sectors:
            raise ValueError("Please select at least one sector for portfolio recommendation.")
        candidates = []
        skipped_companies = []
        for sector in normalized_sectors:
            for company in self._sector_candidate_universe(sector, max_count=8):
                analysis = self._safe_analyze_company(
                    company,
                    budget,
                    holding_period,
                    prefer_live_data=True,
                    use_llm=False,
                    force_refresh_inputs=True,
                    primary_horizon_days=horizon_days,
                    include_short_term_signal=True,
                )
                if analysis is None:
                    skipped_companies.append(company["ticker"])
                    continue
                candidates.append(analysis)
        if not candidates:
            raise ValueError("No eligible stocks with sufficient 10-year history were available for the selected sectors.")
        ranked_candidates = sorted(
            candidates,
            key=lambda item: self._portfolio_score(item, normalized_risk, holding_period, budget),
            reverse=True,
        )
        target_portfolio_size = self._target_portfolio_size(budget, normalized_risk)
        ranking = [item for item in ranked_candidates if self._is_affordable_candidate(item, budget)][:target_portfolio_size]
        if len(ranking) < target_portfolio_size:
            remaining = [item for item in ranked_candidates if item not in ranking]
            ranking.extend(remaining[: target_portfolio_size - len(ranking)])
        allocation_weights = self._allocation_weights(ranking)
        recommendations = []
        for item, weight in zip(ranking, allocation_weights):
            projected_ratio = self._projected_ratio(item["prediction"])
            projected_ratio_5d = self._projected_ratio_from_price(
                item["prediction"].get("predicted_price_5d"),
                item["prediction"].get("current_price"),
            )
            recommendations.append(
                {
                    "company": item["company"]["company"],
                    "ticker": item["company"]["ticker"],
                    "sector": item["company"]["sector"],
                    "current_price": item["prediction"]["current_price"],
                    "predicted_price": item["prediction"]["predicted_price"],
                    "historical_reference_price": item["prediction"]["historical_reference_price"],
                    "growth_pct": item["prediction"]["predicted_growth_pct"],
                    "growth_probability": item["prediction"]["growth_probability"],
                    "allocation_pct": round(weight * 100, 2),
                    "allocation_amount": round(budget * weight, 2),
                    "holding_time_days": horizon_days,
                    "risk_level": item["prediction"]["risk_level"],
                    "quote_source": item["prediction"].get("quote", {}).get("quote_source", "historical_cache"),
                    "live_price_available": bool(item["prediction"].get("quote", {}).get("is_live", False)),
                    "projected_ratio": projected_ratio,
                    "projected_ratio_5d": projected_ratio_5d,
                    "short_term_signal_pct": item["prediction"].get("short_term_signal_pct", 0.0),
                    "recommendation": item["strategy"]["recommendation"],
                    "prediction_horizon_days": item["prediction"].get("prediction_horizon_days", horizon_days),
                    "explanation": item["strategy"]["explanation"],
                }
            )
        recommendations = self._refresh_live_quotes_for_shortlist(recommendations)
        export_name = self._export_to_excel("portfolio", recommendations)
        result = {
            "portfolio": recommendations,
            "excel_file": self._download_payload(export_name),
            "summary": {
                "generated_at": datetime.utcnow().isoformat(),
                "budget": budget,
                "risk_tolerance": normalized_risk,
                "holding_period": holding_period,
                "prediction_horizon_days": horizon_days,
                "selected_sectors": normalized_sectors,
                "skipped_companies": skipped_companies,
                "average_growth_pct": round(sum(item["growth_pct"] for item in recommendations) / max(1, len(recommendations)), 2),
            },
        }
        return result

    def sector_analysis(self, sector: str) -> Dict:
        if sector not in SECTORS:
            raise ValueError(f"Sector '{sector}' is not supported.")
        analyses = []
        skipped_companies = []
        for company in self._sector_candidate_universe(sector, max_count=12):
            analysis = self._safe_analyze_company(
                company,
                100000,
                "medium term",
                prefer_live_data=True,
                use_llm=False,
                force_refresh_inputs=True,
                primary_horizon_days=30,
                include_sector_trend=True,
            )
            if analysis is None:
                skipped_companies.append(company["ticker"])
                continue
            analyses.append(analysis)
        if not analyses:
            raise ValueError(f"No eligible stocks with sufficient 10-year history were available in sector '{sector}'.")
        best = sorted(analyses, key=lambda item: item["ranking_score"], reverse=True)[:6]
        recommendations = [
            {
                "company": item["company"]["company"],
                "ticker": item["company"]["ticker"],
                "sector": item["company"]["sector"],
                "current_price": item["prediction"]["current_price"],
                "predicted_price": item["prediction"]["predicted_price"],
                "historical_reference_price": item["prediction"]["historical_reference_price"],
                "growth_pct": item["prediction"]["predicted_growth_pct"],
                "growth_probability": item["prediction"]["growth_probability"],
                "allocation_pct": round(100 / 6, 2),
                "allocation_amount": 0.0,
                "holding_time_days": 30,
                "risk_level": item["prediction"]["risk_level"],
                "quote_source": item["prediction"].get("quote", {}).get("quote_source", "historical_cache"),
                "live_price_available": bool(item["prediction"].get("quote", {}).get("is_live", False)),
                "projected_ratio": self._projected_ratio(item["prediction"]),
                "projected_ratio_90d": self._projected_ratio_from_price(
                    item["prediction"].get("predicted_price_90d"),
                    item["prediction"].get("current_price"),
                ),
                "growth_90d": item["prediction"].get("growth_90d", 0.0),
                "trend_90d": item["prediction"].get("trend_90d", "Neutral"),
                "recommendation": item["strategy"]["recommendation"],
                "prediction_horizon_days": 30,
                "explanation": item["strategy"]["explanation"],
            }
            for item in best
        ]
        recommendations = self._refresh_live_quotes_for_shortlist(recommendations)
        average_growth_90d = round(sum(item.get("growth_90d", 0.0) for item in recommendations) / max(1, len(recommendations)), 2)
        export_name = self._export_to_excel("sector_analysis", recommendations)
        result = {
            "sector": sector,
            "recommendations": recommendations,
            "excel_file": self._download_payload(export_name),
            "summary": {
                "generated_at": datetime.utcnow().isoformat(),
                "sector": sector,
                "skipped_companies": skipped_companies,
                "average_growth_pct": round(sum(item["growth_pct"] for item in recommendations) / max(1, len(recommendations)), 2),
                "growth_90d": average_growth_90d,
                "trend_90d": self._trend_from_growth(average_growth_90d),
            },
        }
        return result

    def company_analysis(self, company_keyword: str, total_budget: float | None = None, days: int = 30) -> Dict:
        company = self.collector.resolve_company(company_keyword)
        budget = total_budget or 100000
        horizon_days = self._normalize_company_horizon(days)

        prefetch = self._prefetch_realtime_inputs(company, prefer_live_data=True, force_refresh=True)

        analysis = self._analyze_company(
            company,
            budget,
            "medium term",
            prefer_live_data=True,
            use_llm=True,
            prefetched_inputs=prefetch,
            primary_horizon_days=horizon_days,
            include_short_term_signal=True,
        )
        export_name = self._export_to_excel(
            "company_analysis",
            [
                {
                    "company": analysis["company"]["company"],
                    "ticker": analysis["company"]["ticker"],
                    "sector": analysis["company"]["sector"],
                    "current_price": analysis["prediction"]["current_price"],
                    "predicted_price": analysis["prediction"]["predicted_price"],
                    "predicted_growth_pct": analysis["prediction"]["predicted_growth_pct"],
                    "risk_level": analysis["prediction"]["risk_level"],
                    "allocation_pct": analysis["allocation_suggestion_pct"],
                    "holding_time_days": analysis["holding_days"],
                    "recommendation": analysis["strategy"]["recommendation"],
                    "confidence_score": analysis["strategy"]["confidence_score"],
                    "explanation": analysis["strategy"]["explanation"],
                }
            ],
        )
        historical = analysis["stock_frame"][["date", "close"]].tail(180).copy()
        quote = analysis["prediction"].get("quote", {})
        historical["date"] = pd.to_datetime(historical["date"], errors="coerce").dt.tz_localize(None)
        if quote.get("is_live") and quote.get("current_price") is not None:
            quote_dt = pd.to_datetime(quote.get("quote_time"), errors="coerce")
            if pd.notna(quote_dt):
                if getattr(quote_dt, "tzinfo", None) is not None:
                    quote_dt = quote_dt.tz_localize(None)
                live_row = pd.DataFrame([{"date": quote_dt, "close": float(quote["current_price"])}])
                historical = pd.concat([historical, live_row], ignore_index=True)
                historical = historical.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        historical["date"] = historical["date"].dt.strftime("%Y-%m-%d")
        result = {
            "company": analysis["company"],
            "insights": {
                "current_price": analysis["prediction"]["current_price"],
                "historical_reference_price": analysis["prediction"]["historical_reference_price"],
                "predicted_price": analysis["prediction"]["predicted_price"],
                "growth_probability": analysis["prediction"]["growth_probability"],
                "predicted_growth_pct": analysis["prediction"]["predicted_growth_pct"],
                "short_term_signal_pct": analysis["prediction"].get("short_term_signal_pct", 0.0),
                "prediction_horizon_days": analysis["prediction"].get("prediction_horizon_days", horizon_days),
                "risk_level": analysis["prediction"]["risk_level"],
                "quote": analysis["prediction"].get("quote", {}),
                "allocation_suggestion_pct": analysis["allocation_suggestion_pct"],
                "allocation_amount": analysis["allocation_amount"],
                "holding_recommendation": f"{analysis['holding_days']} days",
                "recommendation": analysis["strategy"]["recommendation"],
                "confidence_score": analysis["strategy"]["confidence_score"],
                "explanation": analysis["strategy"]["explanation"],
                "agent_breakdown": analysis["strategy"]["agent_breakdown"],
                "news_analysis": analysis["news_analysis"],
                "macro_news_analysis": analysis["macro_news_analysis"],
                "regulatory_score": analysis["regulatory_score"],
                "policy_score": analysis["policy_score"],
                "policy_drivers": analysis["policy_drivers"],
                "regulatory_drivers": analysis["regulatory_drivers"],
                "macro_drivers": analysis["macro_drivers"],
            },
            "chart": {"historical": historical.to_dict(orient="records"), "prediction": analysis["prediction"]["prediction_series"]},
            "excel_file": self._download_payload(export_name),
        }
        return result

    def backtest_company(self, company_keyword: str, days: int = 5) -> Dict:
        company = self.collector.resolve_company(company_keyword)
        stock_clean = self.preprocessor.clean_stock_data(
            self.collector.fetch_stock_history(company["ticker"], company["sector"], years=10, prefer_live=True)
        )
        if len(stock_clean) < 260:
            raise ValueError("Not enough historical data available for backtesting.")

        allowed_days = {7, 15, 30, 60, 90}
        days = int(days)
        if days not in allowed_days:
            raise ValueError("Backtest days must be one of: 7, 15, 30, 60, 90.")
        rows = []
        for offset in range(days, 0, -1):
            target_idx = len(stock_clean) - offset
            history_slice = stock_clean.iloc[:target_idx].copy()
            if len(history_slice) < 240:
                continue
            enriched = self.preprocessor.attach_context_scores(history_slice, 0.0, 0.0, 0.0, 0.0)
            features = self.feature_engineer.create_features(enriched)
            if features.empty:
                continue

            prediction = self.model_pipeline.predict_backtest(
                frame=features,
                horizon_days=1,
                context_score=0.0,
            )

            previous_close = float(history_slice["close"].iloc[-1])
            actual_close = float(stock_clean.iloc[target_idx]["close"])
            predicted_close = float(prediction["predicted_price"])
            abs_error = abs(predicted_close - actual_close)
            pct_error = abs_error / actual_close * 100 if actual_close else 0.0
            predicted_direction = "up" if predicted_close >= previous_close else "down"
            actual_direction = "up" if actual_close >= previous_close else "down"
            rows.append(
                {
                    "date": pd.to_datetime(stock_clean.iloc[target_idx]["date"]).strftime("%Y-%m-%d"),
                    "previous_close": round(previous_close, 2),
                    "predicted_close": round(predicted_close, 2),
                    "actual_close": round(actual_close, 2),
                    "absolute_error": round(abs_error, 2),
                    "percentage_error": round(pct_error, 2),
                    "predicted_direction": predicted_direction,
                    "actual_direction": actual_direction,
                    "direction_correct": predicted_direction == actual_direction,
                }
            )

        if not rows:
            raise ValueError("Backtest could not be computed for the selected company.")

        backtest_df = pd.DataFrame(rows)
        mape = float(backtest_df["percentage_error"].mean())
        avg_price_prediction_accuracy = max(0.0, 100.0 - mape)
        summary = {
            "days_tested": int(len(backtest_df)),
            "mean_absolute_error": round(float(backtest_df["absolute_error"].mean()), 2),
            "mean_absolute_percentage_error": round(mape, 2),
            "direction_accuracy": round(float(backtest_df["direction_correct"].mean() * 100), 2),
            "average_price_prediction_accuracy": round(avg_price_prediction_accuracy, 2),
        }
        return {
            "company": company,
            "summary": summary,
            "results": backtest_df.to_dict(orient="records"),
            "chart": backtest_df[["date", "predicted_close", "actual_close"]].to_dict(orient="records"),
        }

    def _analyze_company(
        self,
        company: Dict,
        total_budget: float,
        holding_period: str,
        prefer_live_data: bool = True,
        use_llm: bool = True,
        force_refresh_inputs: bool = False,
        prefetched_inputs: Dict | None = None,
        primary_horizon_days: int | None = None,
        include_short_term_signal: bool = False,
        include_sector_trend: bool = False,
    ) -> Dict:
        if prefetched_inputs is None:
            prefetched_inputs = self._prefetch_realtime_inputs(
                company,
                prefer_live_data=prefer_live_data,
                force_refresh=force_refresh_inputs,
            )
        stock_clean = prefetched_inputs["stock_clean"]
        quote = (
            prefetched_inputs["quote"]
            if prefer_live_data
            else {
                "current_price": None,
                "quote_time": datetime.utcnow().isoformat(),
                "quote_source": "historical_cache",
                "currency": "INR",
                "is_live": False,
                "provider_attempts": [],
            }
        )
        raw_news_items = prefetched_inputs["news_items"]
        macro_news_items = prefetched_inputs["macro_news_items"]
        direct_company_news = [item for item in raw_news_items if self._is_direct_company_news(item, company)]
        effective_news_items = direct_company_news[:]
        news_source = "direct_company_news"
        if not effective_news_items:
            sector_news_items = self.collector.fetch_news(
                company["sector"],
                company["sector"],
                8,
                force_refresh_inputs,
            )
            if sector_news_items:
                effective_news_items = sector_news_items
                news_source = "sector_news_fallback"
            elif macro_news_items:
                effective_news_items = macro_news_items
                news_source = "macro_news_fallback"
            else:
                effective_news_items = raw_news_items
                news_source = "direct_company_news"

        news_analysis = self.sentiment_engine.score_articles(
            effective_news_items,
            company["sector"],
            stream_key=f"company:{company['ticker']}",
        )
        macro_news_analysis = self.sentiment_engine.score_articles(
            macro_news_items,
            company["sector"],
            stream_key=f"macro:{company['sector']}",
        )
        news_analysis["news_source"] = news_source
        news_analysis["direct_company_news_count"] = len(direct_company_news)
        news_analysis["effective_news_count"] = len(effective_news_items)
        news_analysis["used_sector_fallback"] = news_source in {"sector_news_fallback", "macro_news_fallback"}
        sebi_updates = prefetched_inputs["sebi_updates"]
        policy_updates = prefetched_inputs["policy_updates"]
        regulatory_score = self._regulatory_score(sebi_updates)
        policy_score = self._policy_score(policy_updates)
        policy_drivers = self._extract_drivers(policy_updates, source="policy")
        regulatory_drivers = self._extract_drivers(sebi_updates, source="regulation")
        macro_drivers = macro_news_analysis.get("key_drivers", [])
        enriched = self.preprocessor.attach_context_scores(stock_clean, news_analysis["sentiment_score"], news_analysis["sector_impact"], regulatory_score, policy_score)
        features = self.feature_engineer.create_features(enriched)
        context_score = self._context_score(news_analysis, macro_news_analysis, policy_score, regulatory_score)
        context_signature = self._build_context_signature(news_analysis, macro_news_analysis, policy_updates, sebi_updates, policy_score, regulatory_score)
        resolved_horizon_days = int(primary_horizon_days or HOLDING_WINDOWS.get(holding_period.lower(), 30))
        requested_horizons = {resolved_horizon_days}
        if include_short_term_signal:
            requested_horizons.add(5)
        if include_sector_trend:
            requested_horizons.add(90)
        horizon_predictions = self.model_pipeline.predict_multi_horizon(
            company["ticker"],
            features,
            sorted(requested_horizons),
            live_price=quote.get("current_price"),
            quote_meta=quote,
            context_score=context_score,
            context_signature=context_signature,
        )
        prediction = dict(horizon_predictions[resolved_horizon_days])
        current_price = float(prediction["current_price"])
        prediction["predicted_growth_pct"] = self._growth_pct(prediction["predicted_price"], current_price)
        prediction["prediction_horizon_days"] = resolved_horizon_days
        if include_short_term_signal and 5 in horizon_predictions:
            prediction["predicted_price_5d"] = horizon_predictions[5]["predicted_price"]
            prediction["short_term_signal_pct"] = self._growth_pct(horizon_predictions[5]["predicted_price"], current_price)
        if include_sector_trend and 90 in horizon_predictions:
            prediction["predicted_price_90d"] = horizon_predictions[90]["predicted_price"]
            prediction["growth_90d"] = self._growth_pct(horizon_predictions[90]["predicted_price"], current_price)
            prediction["trend_90d"] = self._trend_from_growth(prediction["growth_90d"])
        strategy = self.agent_system.run(
            {
                "company": company,
                "prediction": prediction,
                "news_analysis": news_analysis,
                "macro_news_analysis": macro_news_analysis,
                "regulatory_score": regulatory_score,
                "policy_score": policy_score,
                "policy_drivers": policy_drivers,
                "regulatory_drivers": regulatory_drivers,
                "macro_drivers": macro_drivers,
                "news_source": news_source,
                "use_llm": use_llm,
            }
        )
        allocation_pct = self._allocation_pct(prediction, strategy["recommendation"])
        ranking_score = (
            0.4 * prediction["growth_probability"]
            + 0.2 * max(news_analysis["sentiment_score"], 0)
            + 0.15 * max(news_analysis["sector_impact"], 0)
            + 0.15 * max(policy_score, 0)
            + 0.1 * strategy["confidence_score"]
        )
        return {
            "company": company,
            "stock_frame": stock_clean,
            "prediction": prediction,
            "news_analysis": news_analysis,
            "macro_news_analysis": macro_news_analysis,
            "regulatory_score": regulatory_score,
            "policy_score": policy_score,
            "policy_drivers": policy_drivers,
            "regulatory_drivers": regulatory_drivers,
            "macro_drivers": macro_drivers,
            "strategy": strategy,
            "ranking_score": ranking_score,
            "allocation_suggestion_pct": allocation_pct,
            "allocation_amount": round(total_budget * allocation_pct / 100, 2),
            "holding_days": resolved_horizon_days,
        }

    def _safe_analyze_company(
        self,
        company: Dict,
        total_budget: float,
        holding_period: str,
        prefer_live_data: bool = True,
        use_llm: bool = True,
        force_refresh_inputs: bool = False,
        primary_horizon_days: int | None = None,
        include_short_term_signal: bool = False,
        include_sector_trend: bool = False,
    ) -> Dict | None:
        try:
            return self._analyze_company(
                company,
                total_budget,
                holding_period,
                prefer_live_data=prefer_live_data,
                use_llm=use_llm,
                force_refresh_inputs=force_refresh_inputs,
                primary_horizon_days=primary_horizon_days,
                include_short_term_signal=include_short_term_signal,
                include_sector_trend=include_sector_trend,
            )
        except Exception:
            return None

    def _prefetch_realtime_inputs(self, company: Dict, prefer_live_data: bool = True, force_refresh: bool = False) -> Dict:
        with ThreadPoolExecutor(max_workers=6) as executor:
            stock_future = executor.submit(
                self.collector.fetch_stock_history,
                company["ticker"],
                company["sector"],
                10,
                True,
            )
            news_future = executor.submit(self.collector.fetch_news, company["company"], company["sector"], 8, force_refresh)
            macro_news_future = executor.submit(self.collector.fetch_macro_news, company["sector"], 8, force_refresh)
            sebi_future = executor.submit(self.collector.fetch_sebi_updates)
            policy_future = executor.submit(self.collector.fetch_policy_updates, company["sector"], 8, force_refresh)
            quote_future = executor.submit(self.collector.fetch_live_quote, company["ticker"]) if prefer_live_data else None

            stock_clean = self.preprocessor.clean_stock_data(stock_future.result())
            return {
                "stock_clean": stock_clean,
                "quote": quote_future.result() if quote_future is not None else {
                    "current_price": None,
                    "quote_time": datetime.utcnow().isoformat(),
                    "quote_source": "historical_cache",
                    "currency": "INR",
                    "is_live": False,
                    "provider_attempts": [],
                },
                "news_items": news_future.result(),
                "macro_news_items": macro_news_future.result(),
                "sebi_updates": sebi_future.result(),
                "policy_updates": policy_future.result(),
            }

    def _should_bypass_company_cache(self, stock_clean: pd.DataFrame, quote: Dict) -> bool:
        recent = stock_clean["close"].tail(5)
        if len(recent) < 5:
            return False
        baseline = float(recent.iloc[0])
        latest = float(quote.get("current_price")) if quote.get("current_price") is not None else float(recent.iloc[-1])
        if baseline == 0:
            return False
        return abs((latest / baseline) - 1) >= 0.025

    def _get_repeated_company_result(self, ticker: str, budget: float, days: int):
        if (
            LAST_COMPANY_REQUEST.get("company") == ticker
            and LAST_COMPANY_REQUEST.get("budget") == round(float(budget), 2)
            and LAST_COMPANY_REQUEST.get("days") == int(days)
            and is_cache_valid(float(LAST_COMPANY_REQUEST.get("timestamp", 0.0)), REPEATED_REQUEST_TTL_SECONDS)
        ):
            return copy.deepcopy(LAST_COMPANY_REQUEST.get("result"))
        return None

    def _remember_company_request(self, ticker: str, budget: float, days: int, result: Dict) -> None:
        LAST_COMPANY_REQUEST["company"] = ticker
        LAST_COMPANY_REQUEST["budget"] = round(float(budget), 2)
        LAST_COMPANY_REQUEST["days"] = int(days)
        LAST_COMPANY_REQUEST["timestamp"] = time.time()
        LAST_COMPANY_REQUEST["result"] = copy.deepcopy(result)

    def _context_score(self, news_analysis: Dict, macro_news_analysis: Dict, policy_score: float, regulatory_score: float) -> float:
        score = (
            0.42 * news_analysis.get("sentiment_score", 0.0)
            + 0.18 * news_analysis.get("sector_impact", 0.0)
            + 0.20 * macro_news_analysis.get("sentiment_score", 0.0)
            + 0.10 * macro_news_analysis.get("sector_impact", 0.0)
            + 0.06 * policy_score
            + 0.04 * regulatory_score
        )
        return round(score, 4)

    def _build_context_signature(
        self,
        news_analysis: Dict,
        macro_news_analysis: Dict,
        policy_updates: List[Dict],
        sebi_updates: List[Dict],
        policy_score: float,
        regulatory_score: float,
    ) -> Dict:
        latest_policy = policy_updates[0].get("published", "") if policy_updates else ""
        latest_regulation = sebi_updates[0].get("published", "") if sebi_updates else ""
        latest_news = ""
        article_analysis = news_analysis.get("article_analysis", [])
        if article_analysis:
            latest_news = article_analysis[0].get("published", "")
        latest_macro = ""
        macro_article_analysis = macro_news_analysis.get("article_analysis", [])
        if macro_article_analysis:
            latest_macro = macro_article_analysis[0].get("published", "")
        return {
            "news_score": round(float(news_analysis.get("sentiment_score", 0.0)), 4),
            "news_impact": round(float(news_analysis.get("sector_impact", 0.0)), 4),
            "macro_score": round(float(macro_news_analysis.get("sentiment_score", 0.0)), 4),
            "macro_impact": round(float(macro_news_analysis.get("sector_impact", 0.0)), 4),
            "policy_score": round(float(policy_score), 4),
            "regulatory_score": round(float(regulatory_score), 4),
            "news_items": len(article_analysis),
            "macro_items": len(macro_article_analysis),
            "policy_items": len(policy_updates),
            "regulation_items": len(sebi_updates),
            "latest_news": latest_news,
            "latest_macro": latest_macro,
            "latest_policy": latest_policy,
            "latest_regulation": latest_regulation,
        }

    def _projected_ratio(self, prediction: Dict) -> float:
        current = prediction.get("current_price") or 0
        target = prediction.get("predicted_price") or 0
        return float(target / current) if current else 1.0

    def _sector_candidate_universe(self, sector: str, max_count: int = 10) -> List[Dict]:
        catalog = self.collector.get_company_catalog()
        sector_df = catalog[catalog["sector"] == sector].copy()
        if sector_df.empty:
            return []
        sector_df = sector_df.sort_values(["is_leader", "company"], ascending=[False, True]).head(max_count)
        return sector_df.to_dict(orient="records")

    def _normalize_selected_sectors(self, selected_sectors: List[str] | None) -> List[str]:
        if selected_sectors is None:
            return list(SECTORS)
        cleaned = []
        for sector in selected_sectors:
            if sector in SECTORS and sector not in cleaned:
                cleaned.append(sector)
        invalid = [sector for sector in selected_sectors if sector not in SECTORS]
        if invalid:
            raise ValueError(f"Unsupported sector selection: {', '.join(invalid)}")
        return cleaned

    def _portfolio_cache_key(
        self,
        budget: float,
        risk_tolerance: str,
        holding_period: str,
        selected_sectors: List[str] | None,
    ) -> str:
        sectors = self._normalize_selected_sectors(selected_sectors)
        return f"portfolio:{round(float(budget), 2)}:{risk_tolerance.strip().lower()}:{holding_period.strip().lower()}:{'|'.join(sectors)}"

    def _holding_period_to_horizon(self, holding_period: str) -> int:
        mapping = {
            "short term": 15,
            "medium term": 30,
            "long term": 90,
        }
        return mapping.get(holding_period.strip().lower(), 30)

    def _normalize_company_horizon(self, days: int) -> int:
        allowed = {15, 30, 90}
        normalized = int(days)
        if normalized not in allowed:
            raise ValueError("Company prediction horizon must be one of: 15, 30, 90.")
        return normalized

    def _growth_pct(self, predicted_price: float, current_price: float) -> float:
        if not current_price:
            return 0.0
        return round(((float(predicted_price) - float(current_price)) / float(current_price)) * 100, 2)

    def _projected_ratio_from_price(self, predicted_price: float | None, current_price: float | None) -> float:
        if not predicted_price or not current_price:
            return 1.0
        return float(predicted_price) / float(current_price)

    def _trend_from_growth(self, growth_90d: float) -> str:
        if growth_90d > 5:
            return "Bullish"
        if growth_90d < -5:
            return "Bearish"
        return "Neutral"

    def _portfolio_score(self, item: Dict, risk_tolerance: str, holding_period: str, budget: float) -> float:
        prediction = item["prediction"]
        news = item["news_analysis"]
        growth_pct = float(prediction.get("predicted_growth_pct", 0.0))
        growth_prob = float(prediction.get("growth_probability", 0.0))
        current_price = float(prediction.get("current_price", 0.0))
        volatility = float(prediction.get("realized_volatility_30d", 0.0))
        drawdown = abs(float(prediction.get("recent_drawdown_90d", 0.0)))
        context_score = float(prediction.get("context_score", 0.0))
        base = (
            0.34 * growth_prob
            + 0.26 * (growth_pct / 100.0)
            + 0.14 * max(news.get("sentiment_score", 0.0), 0.0)
            + 0.10 * max(news.get("sector_impact", 0.0), 0.0)
            + 0.08 * max(item.get("policy_score", 0.0), 0.0)
            + 0.08 * max(context_score, 0.0)
        )

        if risk_tolerance == "low":
            base -= 0.42 * volatility
            base -= 0.22 * drawdown
            base -= {"high": 0.22, "mid": 0.08, "low": 0.0}.get(prediction["risk_level"], 0.08)
        elif risk_tolerance == "mid":
            base -= 0.20 * volatility
            base -= 0.10 * drawdown
            base -= {"high": 0.09, "mid": 0.03, "low": 0.0}.get(prediction["risk_level"], 0.03)
        else:
            base += 0.06 * (growth_pct / 100.0)
            base -= 0.05 * volatility
            base -= {"high": 0.01, "mid": 0.0, "low": 0.0}.get(prediction["risk_level"], 0.0)

        holding = holding_period.strip().lower()
        if holding == "short term":
            base += 0.16 * max(news.get("sentiment_score", 0.0), 0.0)
            base += 0.10 * max(context_score, 0.0)
            base -= 0.16 * drawdown
            base -= 0.10 * volatility
        elif holding == "long term":
            base += 0.14 * max(item.get("policy_score", 0.0), 0.0)
            base += 0.12 * max(item.get("regulatory_score", 0.0), 0.0)
            base += 0.08 * max(news.get("sector_impact", 0.0), 0.0)
        else:
            base += 0.05 * max(item.get("policy_score", 0.0), 0.0)

        if current_price > budget:
            base -= 0.40
        elif current_price > budget * 0.35:
            base -= 0.15
        elif current_price < max(budget * 0.03, 5000):
            base += 0.03

        return float(base)

    def _is_affordable_candidate(self, item: Dict, budget: float) -> bool:
        current_price = float(item["prediction"].get("current_price", 0.0))
        return current_price <= max(budget * 0.4, 1.0)

    def _target_portfolio_size(self, budget: float, risk_tolerance: str) -> int:
        if budget < 25_000:
            base = 4
        elif budget < 75_000:
            base = 6
        elif budget < 150_000:
            base = 8
        elif budget < 300_000:
            base = 10
        elif budget < 600_000:
            base = 12
        else:
            base = 15

        if risk_tolerance == "low":
            base += 1
        elif risk_tolerance == "high":
            base -= 1

        return max(4, min(15, base))

    def _refresh_live_quotes_for_shortlist(self, recommendations: List[Dict]) -> List[Dict]:
        refreshed = []
        for item in recommendations:
            quote = self.collector.fetch_live_quote(item["ticker"])
            current_price = item["current_price"]
            predicted_price = item["predicted_price"]
            if quote.get("is_live") and quote.get("current_price") is not None:
                current_price = round(float(quote["current_price"]), 2)
                predicted_price = round(current_price * item.get("projected_ratio", 1.0), 2)
                growth_pct = round((predicted_price / current_price - 1) * 100, 2) if current_price else item["growth_pct"]
                item["growth_pct"] = growth_pct
                if "projected_ratio_5d" in item:
                    predicted_price_5d = round(current_price * item.get("projected_ratio_5d", 1.0), 2)
                    item["short_term_signal_pct"] = self._growth_pct(predicted_price_5d, current_price)
                if "projected_ratio_90d" in item:
                    predicted_price_90d = round(current_price * item.get("projected_ratio_90d", 1.0), 2)
                    item["growth_90d"] = self._growth_pct(predicted_price_90d, current_price)
                    item["trend_90d"] = self._trend_from_growth(item["growth_90d"])
            item["current_price"] = current_price
            item["predicted_price"] = predicted_price
            item["quote_source"] = quote.get("quote_source", "historical_cache")
            item["live_price_available"] = bool(quote.get("is_live", False))
            item.pop("projected_ratio", None)
            item.pop("projected_ratio_5d", None)
            item.pop("projected_ratio_90d", None)
            refreshed.append(item)
        return refreshed

    def _extract_drivers(self, items: List[Dict], source: str) -> List[Dict]:
        keyword_map = {
            "war": "war and geopolitical tension",
            "geopolitical": "war and geopolitical tension",
            "investment": "heavy investment in the sector",
            "capex": "heavy capex and investment cycle",
            "infrastructure": "infrastructure spending",
            "policy": "policy support",
            "incentive": "government incentives",
            "export": "export demand",
            "procurement": "defence procurement",
            "defence": "defence spending",
            "inflation": "raw material inflation",
            "margin pressure": "margin pressure",
            "compliance": "compliance and regulatory pressure",
            "governance": "governance and disclosure norms",
            "digital": "digital adoption",
            "ai": "AI adoption and technology spending",
            "energy transition": "energy transition investment",
            "rural demand": "rural demand recovery",
            "healthcare spending": "healthcare spending growth",
        }
        driver_scores: Dict[str, int] = {}
        for item in items:
            text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
            for key, label in keyword_map.items():
                if key in text:
                    driver_scores[label] = driver_scores.get(label, 0) + 1
        ordered = sorted(driver_scores.items(), key=lambda pair: pair[1], reverse=True)
        return [{"label": label, "source": source, "count": count} for label, count in ordered[:4]]

    def _is_direct_company_news(self, article: Dict, company: Dict) -> bool:
        text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
        if not text.strip():
            return False
        company_name = str(company.get("company", "")).strip().lower()
        ticker = str(company.get("ticker", "")).strip().lower().replace(".ns", "")
        if company_name and company_name in text:
            return True
        if ticker and ticker in text:
            return True
        keywords = [token for token in company_name.replace("&", " ").replace("-", " ").split() if len(token) >= 4]
        if not keywords:
            return False
        matches = sum(1 for token in keywords if token in text)
        return matches >= 2

    def _export_to_excel(self, prefix: str, rows: List[Dict]) -> str:
        filename = f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
        pd.DataFrame(rows).to_excel(settings.outputs_dir / filename, index=False)
        return filename

    def _download_payload(self, filename: str) -> Dict:
        return {"filename": filename, "absolute_path": str(settings.outputs_dir / filename)}

    def _regulatory_score(self, regulations: List[Dict]) -> float:
        score = 0.0
        for item in regulations[:8]:
            impact = item.get("impact", "neutral")
            if impact == "positive":
                score += 0.08
            elif impact == "negative":
                score -= 0.08
        return round(score / max(1, min(8, len(regulations))), 4)

    def _policy_score(self, policies: List[Dict]) -> float:
        score = 0.0
        for item in policies[:6]:
            impact = item.get("impact", "neutral")
            if impact == "positive":
                score += 0.1
            elif impact == "negative":
                score -= 0.1
        return round(score / max(1, min(6, len(policies))), 4)

    def _allocation_pct(self, prediction: Dict, recommendation: str) -> float:
        normalized = str(recommendation).strip().lower()
        if normalized in {"sell", "avoid"}:
            base = 3.0
        elif normalized == "hold":
            base = 7.5
        else:
            base = 10.0
        adjustment = 2.0 if prediction["risk_level"] == "low" else 0.0 if prediction["risk_level"] == "mid" else -2.5
        return round(max(2.5, min(15.0, base + adjustment)), 2)

    def _allocation_weights(self, ranking: List[Dict]) -> List[float]:
        scores = [max(item["ranking_score"], 0.01) for item in ranking]
        total = sum(scores)
        return [score / total for score in scores]

    def _risk_penalty(self, predicted_risk: str, requested_risk: str) -> float:
        if requested_risk == "high":
            return 0.0
        if requested_risk == "mid":
            return 0.02 if predicted_risk == "high" else 0.0
        return 0.05 if predicted_risk == "high" else 0.02 if predicted_risk == "mid" else 0.0
