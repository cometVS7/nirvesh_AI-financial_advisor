from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

from backend.config import HISTORICAL_REFRESH_HOURS, NEWS_REFRESH_HOURS, POLICY_REFRESH_DAYS, REGULATION_REFRESH_DAYS, SECTORS, ensure_directories, settings
from backend.service import FinancialAdvisorService


class TrainingPipeline:
    def __init__(self) -> None:
        ensure_directories()
        self.service = FinancialAdvisorService()

    def refresh_all(self) -> Dict:
        catalog = self.service.collector.get_company_catalog()
        summary: Dict[str, object] = {
            "generated_at": datetime.utcnow().isoformat(),
            "stock_refresh_hours": HISTORICAL_REFRESH_HOURS,
            "news_refresh_hours": NEWS_REFRESH_HOURS,
            "policy_refresh_days": POLICY_REFRESH_DAYS,
            "regulation_refresh_days": REGULATION_REFRESH_DAYS,
            "total_companies": int(len(catalog)),
            "trained_companies": 0,
            "skipped_companies": 0,
            "skips": [],
            "sector_status": {},
        }

        sebi_updates = self.service.collector.fetch_sebi_updates()
        regulatory_score = self.service._regulatory_score(sebi_updates)

        sector_cache: Dict[str, Dict] = {}
        for sector in SECTORS:
            macro_news = self.service.collector.fetch_macro_news(sector)
            macro_news_analysis = self.service.sentiment_engine.score_articles(macro_news, sector)
            policy_updates = self.service.collector.fetch_policy_updates(sector)
            policy_score = self.service._policy_score(policy_updates)
            sector_cache[sector] = {
                "macro_news_analysis": macro_news_analysis,
                "policy_updates": policy_updates,
                "policy_score": policy_score,
            }
            summary["sector_status"][sector] = {
                "macro_articles": len(macro_news),
                "policy_updates": len(policy_updates),
                "policy_score": policy_score,
                "regulatory_score": regulatory_score,
            }

        for company in catalog.to_dict(orient="records"):
            sector = company["sector"]
            try:
                stock_clean = self.service.preprocessor.clean_stock_data(
                    self.service.collector.fetch_stock_history(company["ticker"], sector, years=10, prefer_live=True)
                )
                news_items = self.service.collector.fetch_news(company["company"], sector)
                news_analysis = self.service.sentiment_engine.score_articles(news_items, sector)
                macro_news_analysis = sector_cache[sector]["macro_news_analysis"]
                policy_updates = sector_cache[sector]["policy_updates"]
                policy_score = sector_cache[sector]["policy_score"]

                enriched = self.service.preprocessor.attach_context_scores(
                    stock_clean,
                    news_analysis.get("sentiment_score", 0.0),
                    news_analysis.get("sector_impact", 0.0),
                    regulatory_score,
                    policy_score,
                )
                features = self.service.feature_engineer.create_features(enriched)
                if features.empty:
                    raise ValueError("Feature engineering returned no rows.")

                context_signature = self.service._build_context_signature(
                    news_analysis,
                    macro_news_analysis,
                    policy_updates,
                    sebi_updates,
                    policy_score,
                    regulatory_score,
                )
                artifacts = self.service.model_pipeline.train_or_load(
                    company["ticker"],
                    features,
                    context_signature=context_signature,
                )

                summary["trained_companies"] = int(summary["trained_companies"]) + 1
                summary["sector_status"][sector].setdefault("trained_tickers", []).append(
                    {
                        "ticker": company["ticker"],
                        "company": company["company"],
                        "rows": int(len(features)),
                        "latest_date": str(features["date"].iloc[-1]),
                        "xgboost_mae": artifacts.metrics.get("xgboost_mae"),
                    }
                )
            except Exception as exc:
                summary["skipped_companies"] = int(summary["skipped_companies"]) + 1
                summary["skips"].append(
                    {
                        "ticker": company["ticker"],
                        "company": company["company"],
                        "sector": sector,
                        "error": str(exc),
                    }
                )

        settings.training_manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def main() -> None:
    summary = TrainingPipeline().refresh_all()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
