from __future__ import annotations

from typing import Dict

import numpy as np
from langchain_core.runnables import RunnableLambda, RunnableParallel

from backend.config import settings

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None


class FinancialAgentOrchestrator:
    def __init__(self) -> None:
        self.llm = None
        if settings.gemini_api_key and ChatGoogleGenerativeAI is not None:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=settings.gemini_api_key,
                    temperature=0.2,
                    max_retries=1,
                    timeout=15,
                )
            except Exception:
                self.llm = None
        self.market_agent = RunnableLambda(self._market_agent)
        self.news_agent = RunnableLambda(self._news_agent)
        self.regulation_agent = RunnableLambda(self._regulation_agent)
        self.strategy_agent = RunnableLambda(self._strategy_agent)

    def run(self, context: Dict) -> Dict:
        partial = RunnableParallel(
            market=self.market_agent,
            news_sector=self.news_agent,
            sebi_policy=self.regulation_agent,
        ).invoke(context)
        merged = {
            **context,
            **partial.get("market", {}),
            **partial.get("news_sector", {}),
            **partial.get("sebi_policy", {}),
        }
        return self.strategy_agent.invoke(merged)

    def _market_agent(self, context: Dict) -> Dict:
        prediction = context["prediction"]
        technical_view = "bullish" if prediction["predicted_growth_pct"] > 7 else "accumulate" if prediction["predicted_growth_pct"] > 2 else "cautious"
        return {"market_summary": {"technical_view": technical_view, "predicted_price": prediction["predicted_price"], "growth_probability": prediction["growth_probability"], "risk_level": prediction["risk_level"]}}

    def _news_agent(self, context: Dict) -> Dict:
        news = context["news_analysis"]
        sentiment = news["sentiment_score"]
        outlook = "positive" if sentiment > 0.2 else "negative" if sentiment < -0.15 else "neutral"
        return {"news_summary": {"sentiment_outlook": outlook, "sentiment_score": sentiment, "sector_impact": news["sector_impact"], "key_drivers": news.get("key_drivers", [])}}

    def _regulation_agent(self, context: Dict) -> Dict:
        stance = "supportive" if context["regulatory_score"] + context["policy_score"] >= 0.15 else "watchful"
        return {
            "regulation_summary": {
                "regulatory_score": context["regulatory_score"],
                "policy_score": context["policy_score"],
                "policy_stance": stance,
                "policy_drivers": context.get("policy_drivers", []),
                "regulatory_drivers": context.get("regulatory_drivers", []),
            }
        }

    def _strategy_agent(self, context: Dict) -> Dict:
        prediction = context["prediction"]
        news = context["news_summary"]
        macro_news = context.get("macro_news_analysis", {})
        regulation = context["regulation_summary"]
        growth_pct = float(prediction.get("predicted_growth_pct", 0.0))
        growth_prob = float(prediction.get("growth_probability", 0.5))
        technical_view = context.get("market_summary", {}).get("technical_view", "cautious")
        risk_level = str(prediction.get("risk_level", "mid")).lower()
        realized_volatility = float(prediction.get("realized_volatility_30d", 0.0))
        recent_drawdown = abs(float(prediction.get("recent_drawdown_90d", 0.0)))

        growth_signal = float(np.clip(growth_pct / 12.0, -1.0, 1.0))
        probability_signal = float(np.clip((growth_prob - 0.5) * 2.0, -1.0, 1.0))
        sentiment_signal = float(np.clip(news.get("sentiment_score", 0.0), -1.0, 1.0))
        sector_signal = float(np.clip(news.get("sector_impact", 0.0), -1.0, 1.0))
        macro_sentiment_signal = float(np.clip(macro_news.get("sentiment_score", 0.0), -1.0, 1.0))
        macro_impact_signal = float(np.clip(macro_news.get("sector_impact", 0.0), -1.0, 1.0))
        policy_signal = float(np.clip(regulation.get("policy_score", 0.0), -1.0, 1.0))
        regulatory_signal = float(np.clip(regulation.get("regulatory_score", 0.0), -1.0, 1.0))
        technical_signal = {"bullish": 0.25, "accumulate": 0.08, "cautious": -0.2}.get(technical_view, -0.05)
        risk_signal = {"low": 0.24, "mid": 0.0, "high": -0.28}.get(risk_level, 0.0)
        volatility_signal = float(np.clip((0.022 - realized_volatility) / 0.022, -1.0, 1.0))
        drawdown_signal = float(np.clip((0.15 - recent_drawdown) / 0.15, -1.0, 1.0))

        contribution_weights = {
            "growth_pct": 0.22,
            "growth_probability": 0.20,
            "news_sentiment": 0.13,
            "sector_impact": 0.09,
            "macro_sentiment": 0.08,
            "macro_impact": 0.05,
            "policy_score": 0.08,
            "regulatory_score": 0.04,
            "risk_profile": 0.04,
            "technical_view": 0.03,
            "volatility_regime": 0.02,
            "drawdown_regime": 0.02,
        }
        contribution_inputs = {
            "growth_pct": growth_signal,
            "growth_probability": probability_signal,
            "news_sentiment": sentiment_signal,
            "sector_impact": sector_signal,
            "macro_sentiment": macro_sentiment_signal,
            "macro_impact": macro_impact_signal,
            "policy_score": policy_signal,
            "regulatory_score": regulatory_signal,
            "risk_profile": risk_signal,
            "technical_view": technical_signal,
            "volatility_regime": volatility_signal,
            "drawdown_regime": drawdown_signal,
        }
        contributions = {
            key: round(float(contribution_weights[key] * contribution_inputs[key]), 4)
            for key in contribution_weights
        }
        raw_score = float(sum(contributions.values()))
        confidence_score = float(np.clip(0.5 + raw_score * 0.55, 0.05, 0.95))

        if raw_score >= 0.12:
            recommendation = "buy"
        elif raw_score >= -0.04:
            recommendation = "hold"
        elif raw_score >= -0.20:
            recommendation = "avoid"
        else:
            recommendation = "sell"

        sorted_contrib = sorted(contributions.items(), key=lambda pair: pair[1])
        negative_reasons = [{"factor": k, "value": v} for k, v in sorted_contrib[:3] if v < 0]
        positive_reasons = [{"factor": k, "value": v} for k, v in reversed(sorted_contrib[-3:]) if v > 0]
        return {
            "recommendation": recommendation,
            "confidence_score": round(confidence_score, 4),
            "score_raw": round(raw_score, 4),
            "explanation": self._generate_explanation(context, raw_score, recommendation, contributions),
            "agent_breakdown": {
                "market_analysis_agent": context["market_summary"],
                "news_sector_agent": context["news_summary"],
                "sebi_policy_agent": context["regulation_summary"],
                "score_components": contributions,
                "top_positive_factors": positive_reasons,
                "top_negative_factors": negative_reasons,
            },
        }

    def _generate_explanation(self, context: Dict, score: float, recommendation: str, contributions: Dict[str, float] | None = None) -> str:
        company = context["company"]["company"]
        sector = context["company"]["sector"]
        prediction = context["prediction"]
        news = context["news_analysis"]
        macro_news = context["macro_news_analysis"]
        news_drivers = context.get("news_summary", {}).get("key_drivers", [])
        policy_drivers = context.get("regulation_summary", {}).get("policy_drivers", [])
        regulatory_drivers = context.get("regulation_summary", {}).get("regulatory_drivers", [])
        macro_drivers = context.get("macro_drivers", [])

        company_headlines = [item.get("title", "") for item in news.get("article_analysis", [])[:2] if item.get("title")]
        macro_headlines = [item.get("title", "") for item in macro_news.get("article_analysis", [])[:2] if item.get("title")]
        company_headline_text = " and ".join(f"'{headline}'" for headline in company_headlines) if company_headlines else "no recent company headlines available"
        macro_headline_text = " and ".join(f"'{headline}'" for headline in macro_headlines) if macro_headlines else "no recent macro headlines available"

        driver_text = self._driver_summary(news_drivers, policy_drivers, regulatory_drivers)
        macro_text = self._driver_summary(macro_drivers, [], [])
        upside_view = "strong upside potential" if prediction["predicted_growth_pct"] >= 8 else "moderate upside potential" if prediction["predicted_growth_pct"] >= 3 else "limited upside"
        risk_view = {
            "low": "price behavior has been relatively stable recently",
            "mid": "there is some volatility, but it is not extreme",
            "high": "volatility and drawdown patterns remain elevated",
        }.get(prediction["risk_level"], "risk is balanced")
        recommendation_line = {
            "buy": "The setup supports accumulation at current levels.",
            "hold": "The setup supports a hold or staggered entry rather than an aggressive buy.",
            "avoid": "The setup does not currently justify a fresh position.",
            "sell": "The setup suggests reducing exposure or exiting until conditions improve.",
        }.get(recommendation, "The setup is mixed.")
        quote_source = prediction.get("quote", {}).get("quote_source", "historical")
        explanation_tone = "constructive" if score >= 0.2 else "balanced" if score >= -0.1 else "defensive"
        factor_labels = {
            "growth_pct": "projected growth",
            "growth_probability": "growth probability",
            "news_sentiment": "company-news sentiment",
            "sector_impact": "sector-news impact",
            "macro_sentiment": "macro-news sentiment",
            "macro_impact": "macro-market impact",
            "policy_score": "policy environment",
            "regulatory_score": "regulatory stance",
            "risk_profile": "risk profile",
            "technical_view": "technical setup",
            "volatility_regime": "volatility regime",
            "drawdown_regime": "drawdown regime",
        }
        strongest_positive = ""
        strongest_negative = ""
        strongest_positive_value = 0.0
        strongest_negative_value = 0.0
        if contributions:
            strongest_pos_key, strongest_positive_value = max(contributions.items(), key=lambda item: item[1])
            strongest_neg_key, strongest_negative_value = min(contributions.items(), key=lambda item: item[1])
            strongest_positive = factor_labels.get(strongest_pos_key, strongest_pos_key)
            strongest_negative = factor_labels.get(strongest_neg_key, strongest_neg_key)

        news_freshness = "new company headlines were detected in this run" if news.get("has_new_items") else "headline set is unchanged from the previous scan"
        macro_freshness = "new macro headlines were detected" if macro_news.get("has_new_items") else "macro headline set is unchanged"
        volatility_pct = float(prediction.get("realized_volatility_30d", 0.0)) * 100
        drawdown_pct = abs(float(prediction.get("recent_drawdown_90d", 0.0))) * 100
        decision_basis = {
            "buy": "positive return expectations and supportive context factors dominate downside risk",
            "hold": "signals are mixed, so conviction is moderate and position sizing should stay disciplined",
            "avoid": "downside risk-adjusted return is weak versus available alternatives",
            "sell": "risk and downside pressure currently outweigh recovery signals",
        }.get(recommendation, "signals are mixed")
        news_source = str(context.get("news_source") or news.get("news_source") or "direct_company_news")
        news_source_label = {
            "direct_company_news": "Direct company news",
            "sector_news_fallback": "Sector-linked fallback news",
            "macro_news_fallback": "Macro fallback news",
        }.get(news_source, "News feed")
        direct_count = int(news.get("direct_company_news_count", 0))
        effective_count = int(news.get("effective_news_count", len(news.get("article_analysis", []))))

        return (
            f"Summary\n"
            f"- Recommendation: {recommendation.upper()} ({explanation_tone} conviction, score {score:.2f})\n"
            f"- Price Outlook: Rs. {prediction['current_price']:.2f} -> Rs. {prediction['predicted_price']:.2f} "
            f"({prediction['predicted_growth_pct']:.2f}% for selected horizon)\n"
            f"- Probability & Risk: growth probability {prediction['growth_probability']:.2f}, risk level {prediction['risk_level']}\n\n"
            f"Real-Time Inputs Used\n"
            f"- Live price source: {quote_source}\n"
            f"- {news_source_label}: {effective_count} articles used (direct company matches: {direct_count})\n"
            f"- Company sentiment {news.get('sentiment_score', 0.0):.2f}, sector impact {news.get('sector_impact', 0.0):.2f}\n"
            f"- News freshness: {news_freshness}; Macro freshness: {macro_freshness}\n\n"
            f"Market & Policy Context\n"
            f"- Company headline cues: {company_headline_text}\n"
            f"- Macro headline cues: {macro_headline_text}\n"
            f"- Macro drivers: {macro_text}\n"
            f"- Policy/regulation drivers: {driver_text} "
            f"(policy {context.get('policy_score', 0.0):.2f}, regulatory {context.get('regulatory_score', 0.0):.2f})\n\n"
            f"Risk Diagnostics\n"
            f"- Volatility (30d): {volatility_pct:.2f}% | Drawdown (90d): {drawdown_pct:.2f}%\n"
            f"- Interpretation: {risk_view}\n\n"
            f"Why This Recommendation\n"
            f"- Strongest positive factor: {strongest_positive or 'overall momentum'} (+{strongest_positive_value:.3f})\n"
            f"- Strongest negative factor: {strongest_negative or 'risk-adjusted uncertainty'} ({strongest_negative_value:.3f})\n"
            f"- Decision basis: {decision_basis}\n"
            f"- Action note: {recommendation_line}"
        )

    def _driver_summary(self, news_drivers: list, policy_drivers: list, regulatory_drivers: list) -> str:
        ordered = []
        for group in (news_drivers, policy_drivers, regulatory_drivers):
            for item in group:
                text = item.get("label") if isinstance(item, dict) else str(item)
                if text and text not in ordered:
                    ordered.append(text)
        return ", ".join(ordered[:5]) if ordered else "no dominant external driver identified"
