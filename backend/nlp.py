from __future__ import annotations

import hashlib
from threading import Lock
from typing import Dict, List

import numpy as np

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None


class FinBertSentimentAnalyzer:
    _classifier = None
    _model_load_attempted = False
    _model_lock = Lock()
    _last_news_hash: str | None = None
    _last_result: Dict | None = None
    _result_by_hash: Dict[str, Dict] = {}

    def __init__(self) -> None:
        self.positive_words = {"growth", "expansion", "support", "improve", "surge", "strong", "approval", "incentive", "profit", "recovery"}
        self.negative_words = {"risk", "pressure", "decline", "concern", "slowdown", "penalty", "warning", "weak", "loss", "delay"}

    def load_model(self):
        if self.__class__._classifier is not None or self.__class__._model_load_attempted or pipeline is None:
            return self.__class__._classifier
        with self.__class__._model_lock:
            if self.__class__._classifier is not None or self.__class__._model_load_attempted:
                return self.__class__._classifier
            self.__class__._model_load_attempted = True
            try:
                self.__class__._classifier = pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")
            except Exception:
                self.__class__._classifier = None
        return self.__class__._classifier

    def score_articles(self, articles: List[Dict], sector: str) -> Dict:
        if not articles:
            return {"sentiment_score": 0.0, "sector_impact": 0.0, "article_analysis": []}
        news_hash = self._article_hash(articles)
        if news_hash == self.__class__._last_news_hash and self.__class__._last_result is not None:
            return self.__class__._last_result.copy()
        if news_hash in self.__class__._result_by_hash:
            cached = self.__class__._result_by_hash[news_hash].copy()
            self.__class__._last_news_hash = news_hash
            self.__class__._last_result = cached.copy()
            return cached

        analyses = []
        scores = []
        sector_scores = []
        driver_scores = {}
        for article in articles:
            text = f"{article.get('title', '')}. {article.get('summary', '')}".strip()
            score, label = self._score_text(text)
            sector_impact = self._sector_impact(text, sector, score)
            for driver in self._extract_drivers(text):
                driver_scores[driver] = driver_scores.get(driver, 0) + 1
            analyses.append(
                {
                    "title": article.get("title", ""),
                    "sentiment_label": label,
                    "sentiment_score": round(score, 4),
                    "sector_impact": round(sector_impact, 4),
                    "published": article.get("published", ""),
                }
            )
            scores.append(score)
            sector_scores.append(sector_impact)
        result = {
            "sentiment_score": float(np.mean(scores)),
            "sector_impact": float(np.mean(sector_scores)),
            "article_analysis": analyses,
            "key_drivers": [{"label": label, "count": count} for label, count in sorted(driver_scores.items(), key=lambda item: item[1], reverse=True)[:5]],
        }
        self.__class__._last_news_hash = news_hash
        self.__class__._last_result = result.copy()
        self.__class__._result_by_hash[news_hash] = result.copy()
        return result

    def _score_text(self, text: str) -> tuple[float, str]:
        classifier = self.load_model()
        if classifier is not None:
            try:
                result = classifier(text[:512])[0]
                label = result["label"].lower()
                score = float(result["score"])
                if "positive" in label:
                    return score, "positive"
                if "negative" in label:
                    return -score, "negative"
                return 0.0, "neutral"
            except Exception:
                pass
        lowered = text.lower()
        pos_hits = sum(word in lowered for word in self.positive_words)
        neg_hits = sum(word in lowered for word in self.negative_words)
        score = (pos_hits - neg_hits) / max(1, pos_hits + neg_hits + 1)
        label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
        return float(score), label

    def _sector_impact(self, text: str, sector: str, sentiment_score: float) -> float:
        keywords = sector.lower().split()
        text_lower = text.lower()
        overlap = sum(keyword in text_lower for keyword in keywords)
        return float(sentiment_score * (1 + 0.15 * overlap))

    def _extract_drivers(self, text: str) -> List[str]:
        lowered = text.lower()
        driver_map = {
            "war": "war and geopolitical tension",
            "geopolitical": "war and geopolitical tension",
            "investment": "heavy investment in the sector",
            "capex": "heavy capex and investment cycle",
            "policy": "policy support",
            "incentive": "government incentives",
            "export": "export demand",
            "defence": "defence spending",
            "procurement": "defence procurement",
            "inflation": "raw material inflation",
            "margin": "margin pressure",
            "digital": "digital adoption",
            "ai": "AI adoption and technology spending",
            "energy transition": "energy transition investment",
            "rural demand": "rural demand recovery",
            "healthcare spending": "healthcare spending growth",
        }
        return [label for key, label in driver_map.items() if key in lowered]

    def _article_hash(self, articles: List[Dict]) -> str:
        payload = "||".join(
            f"{article.get('title', '').strip()}::{article.get('published', '').strip()}"
            for article in articles
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
