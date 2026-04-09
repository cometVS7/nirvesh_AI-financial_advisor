from __future__ import annotations

from typing import Dict

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
        regulation = context["regulation_summary"]
        score = (
            0.45 * prediction["growth_probability"]
            + 0.2 * max(news["sentiment_score"], 0)
            + 0.15 * max(news["sector_impact"], 0)
            + 0.1 * max(regulation["policy_score"], 0)
            + 0.1 * (0.8 if prediction["risk_level"] == "low" else 0.5 if prediction["risk_level"] == "mid" else 0.2)
        )
        recommendation = "buy" if score >= 0.62 else "hold" if score >= 0.42 else "avoid"
        return {
            "recommendation": recommendation,
            "confidence_score": round(float(score), 4),
            "explanation": self._generate_explanation(context, score, recommendation),
            "agent_breakdown": {
                "market_analysis_agent": context["market_summary"],
                "news_sector_agent": context["news_summary"],
                "sebi_policy_agent": context["regulation_summary"],
            },
        }

    def _generate_explanation(self, context: Dict, score: float, recommendation: str) -> str:
        company = context["company"]["company"]
        sector = context["company"]["sector"]
        prediction = context["prediction"]
        news = context["news_analysis"]
        news_drivers = context.get("news_summary", {}).get("key_drivers", [])
        policy_drivers = context.get("regulation_summary", {}).get("policy_drivers", [])
        regulatory_drivers = context.get("regulation_summary", {}).get("regulatory_drivers", [])
        macro_drivers = context.get("macro_drivers", [])
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
        }.get(recommendation, "The setup is mixed.")
        quote_source = prediction.get("quote", {}).get("quote_source", "historical")
        return (
            f"{company} currently shows {upside_view}, with a projected move from Rs. {prediction['current_price']:.2f} "
            f"to around Rs. {prediction['predicted_price']:.2f}, implying {prediction['predicted_growth_pct']:.2f}% expected growth. "
            f"The recommendation is '{recommendation}' because growth probability is {prediction['growth_probability']:.2f}, "
            f"while risk is assessed as {prediction['risk_level']} since {risk_view}. "
            f"Company and sector news are being driven by {driver_text}, and the broader macro backdrop is influenced by {macro_text}. "
            f"{recommendation_line} Live quote source used: {quote_source}."
        )

    def _driver_summary(self, news_drivers: list, policy_drivers: list, regulatory_drivers: list) -> str:
        ordered = []
        for group in (news_drivers, policy_drivers, regulatory_drivers):
            for item in group:
                text = item.get("label") if isinstance(item, dict) else str(item)
                if text and text not in ordered:
                    ordered.append(text)
        return ", ".join(ordered[:5]) if ordered else "no dominant external driver identified"
