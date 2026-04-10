from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import feedparser
import pandas as pd
import requests

from backend.config import (
    HISTORICAL_REFRESH_HOURS,
    HISTORICAL_TICKER_ALIASES,
    NEWS_REFRESH_HOURS,
    POLICY_REFRESH_DAYS,
    REGULATION_REFRESH_DAYS,
    build_company_catalog,
    ensure_directories,
    settings,
)

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


class DataCollector:
    def __init__(self) -> None:
        ensure_directories()
        self.catalog = build_company_catalog()
        if yf is not None:
            try:
                yf.set_tz_cache_location(str(settings.stock_cache_dir / "tz_cache"))
            except Exception:
                pass

    def get_company_catalog(self) -> pd.DataFrame:
        return self.catalog.copy()

    def resolve_company(self, keyword: str) -> Dict[str, str]:
        keyword = keyword.strip().lower()
        matches = self.catalog[
            self.catalog["company"].str.lower().str.contains(keyword)
            | self.catalog["ticker"].str.lower().str.contains(keyword)
        ]
        if matches.empty:
            raise ValueError(f"No company found for keyword '{keyword}'.")
        return matches.sort_values(["is_leader", "company"], ascending=[False, True]).iloc[0].to_dict()

    def fetch_stock_history(self, ticker: str, sector: str, years: int = 10, prefer_live: bool = True) -> pd.DataFrame:
        cache_path = settings.stock_cache_dir / f"{ticker.replace('.', '_')}.csv"
        cached = self._load_cached_stock_history(cache_path)
        if cached is not None and self._is_cache_fresh(cache_path, timedelta(hours=HISTORICAL_REFRESH_HOURS)):
            return cached

        data = self._download_with_alias_support(ticker, years) if prefer_live else pd.DataFrame()
        if not data.empty:
            data.to_csv(cache_path, index=False)
            return data
        if cached is not None:
            return cached
        raise ValueError(f"Unable to fetch real {years}-year historical data for {ticker} from Yahoo Finance.")

    def fetch_live_quote(self, ticker: str) -> Dict:
        fallback = {
            "current_price": None,
            "quote_time": datetime.utcnow().isoformat(),
            "quote_source": "live_quote_unavailable",
            "currency": "INR",
            "is_live": False,
            "provider_attempts": [],
        }
        provider_attempts = []
        if yf is None:
            provider_attempts.append({"provider": "yfinance", "status": "unavailable", "detail": "yfinance package not installed"})
            fallback["provider_attempts"] = provider_attempts
            return fallback
        try:
            frame = yf.download(
                ticker,
                period="1d",
                interval="1m",
                auto_adjust=True,
                progress=False,
                threads=False,
                prepost=False,
            )
            if not frame.empty and "Close" in frame.columns:
                latest_price = float(frame["Close"].dropna().iloc[-1])
                latest_dt = frame.index[-1]
                return {
                    "current_price": round(latest_price, 2),
                    "quote_time": pd.Timestamp(latest_dt).isoformat(),
                    "quote_source": "yfinance_intraday_1m",
                    "currency": "INR",
                    "is_live": True,
                    "provider_attempts": [{"provider": "yfinance", "status": "success", "detail": "1d/1m intraday close"}],
                }
            provider_attempts.append({"provider": "yfinance", "status": "empty", "detail": "1d/1m intraday returned no rows"})
        except Exception as exc:
            provider_attempts.append({"provider": "yfinance", "status": "failed", "detail": f"1d/1m intraday failed: {exc}"})
        try:
            ticker_obj = yf.Ticker(ticker)
            recent = ticker_obj.history(period="5d", interval="1d", auto_adjust=True)
            if not recent.empty and "Close" in recent.columns:
                latest_price = float(recent["Close"].dropna().iloc[-1])
                latest_dt = recent.index[-1]
                return {
                    "current_price": round(latest_price, 2),
                    "quote_time": pd.Timestamp(latest_dt).isoformat(),
                    "quote_source": "yfinance_history",
                    "currency": "INR",
                    "is_live": True,
                    "provider_attempts": [{"provider": "yfinance", "status": "success", "detail": "latest history close"}],
                }
            provider_attempts.append({"provider": "yfinance", "status": "empty", "detail": "history returned no rows"})
        except Exception as exc:
            provider_attempts.append({"provider": "yfinance", "status": "failed", "detail": str(exc)})
        try:
            ticker_obj = yf.Ticker(ticker)
            fast_info = getattr(ticker_obj, "fast_info", None)
            if fast_info:
                for key in ("lastPrice", "last_price", "regularMarketPrice", "previousClose"):
                    value = fast_info.get(key)
                    if value is not None:
                        return {
                            "current_price": round(float(value), 2),
                            "quote_time": datetime.utcnow().isoformat(),
                            "quote_source": "yfinance_fast_info",
                            "currency": "INR",
                            "is_live": True,
                            "provider_attempts": [{"provider": "yfinance", "status": "success", "detail": f"fast_info:{key}"}],
                        }
            provider_attempts.append({"provider": "yfinance", "status": "empty", "detail": "fast_info had no usable last price"})
        except Exception:
            provider_attempts.append({"provider": "yfinance", "status": "failed", "detail": "fast_info lookup failed"})
        fallback["provider_attempts"] = provider_attempts
        return fallback

    def fetch_market_mood_snapshot(self) -> Dict:
        cache_path = settings.raw_dir / "market_mood_snapshot.json"
        cached_payload = None
        if cache_path.exists():
            try:
                cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                cached_payload = None

        fallback = cached_payload or {
            "index_symbol": "^NSEI",
            "index_name": "NIFTY 50",
            "mood": "Stable Bearish",
            "trend": "Bearish",
            "volatility_state": "Stable",
            "as_of_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "source": "cache_unavailable",
        }

        if yf is None:
            fallback["source"] = "yfinance_unavailable"
            return fallback

        try:
            frame = yf.download(
                "^NSEI",
                period="6mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if frame.empty:
                return fallback

            if "Close" not in frame.columns:
                return fallback

            closes = frame["Close"].dropna().tail(90)
            if len(closes) < 20:
                return fallback

            last_close = float(closes.iloc[-1])
            sma20 = float(closes.tail(20).mean())
            sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else float(closes.mean())
            pct_5d = float((closes.iloc[-1] / closes.iloc[-6]) - 1) if len(closes) >= 6 else float((closes.iloc[-1] / closes.iloc[0]) - 1)
            returns_20 = closes.pct_change().dropna().tail(20)
            vol_20d = float(returns_20.std()) if len(returns_20) else 0.0

            bullish_score = int(last_close >= sma20) + int(sma20 >= sma50) + int(pct_5d >= 0)
            trend = "Bullish" if bullish_score >= 2 else "Bearish"
            volatility_state = "Volatile" if vol_20d >= 0.014 else "Stable"
            mood_label = f"{volatility_state} {trend}"

            as_of = pd.to_datetime(closes.index[-1], errors="coerce")
            as_of_str = as_of.strftime("%Y-%m-%d") if pd.notna(as_of) else datetime.utcnow().strftime("%Y-%m-%d")

            payload = {
                "index_symbol": "^NSEI",
                "index_name": "NIFTY 50",
                "mood": mood_label,
                "trend": trend,
                "volatility_state": volatility_state,
                "as_of_date": as_of_str,
                "source": "yfinance_daily",
            }
            try:
                cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except Exception:
                pass
            return payload
        except Exception:
            return fallback

    def fetch_news(self, company: str, sector: str, limit: int = 8, force_refresh: bool = False) -> List[Dict]:
        cache_path = settings.news_cache_dir / f"{company.lower().replace(' ', '_')}.json"
        cached = self._load_json_cache(cache_path)
        if cached is not None and not force_refresh:
            fresh_articles = self._download_news(company, sector, min(limit, 3))
            if fresh_articles:
                latest_cached = self._latest_published(cached)
                if any(self._parse_datetime(article.get("published")) > latest_cached for article in fresh_articles if article.get("published")):
                    merged = self._merge_articles(fresh_articles + cached, limit)
                    cache_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
                    return merged
            if self._is_cache_fresh(cache_path, timedelta(hours=NEWS_REFRESH_HOURS)):
                return cached
        articles = self._download_news(company, sector, min(limit, 8))
        if articles:
            cache_path.write_text(json.dumps(articles, indent=2), encoding="utf-8")
            return articles
        return cached or []

    def fetch_macro_news(self, sector: str, limit: int = 8, force_refresh: bool = False) -> List[Dict]:
        cache_path = settings.news_cache_dir / f"macro_{sector.replace(' ', '_')}.json"
        cached = self._load_json_cache(cache_path)
        if cached is not None and not force_refresh:
            fresh_articles = self._download_macro_news(sector, min(limit, 3))
            if fresh_articles:
                latest_cached = self._latest_published(cached)
                if any(self._parse_datetime(article.get("published")) > latest_cached for article in fresh_articles if article.get("published")):
                    merged = self._merge_articles(fresh_articles + cached, limit)
                    cache_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
                    return merged
            if self._is_cache_fresh(cache_path, timedelta(hours=NEWS_REFRESH_HOURS)):
                return cached
        articles = self._download_macro_news(sector, min(limit, 8))
        if articles:
            cache_path.write_text(json.dumps(articles, indent=2), encoding="utf-8")
            return articles
        return cached or []

    def fetch_sebi_updates(self, limit: int = 10) -> List[Dict]:
        cache_path = settings.regulatory_cache_dir / "sebi_updates.json"
        cached = self._load_json_cache(cache_path)
        if cached is not None and self._is_cache_fresh(cache_path, timedelta(days=REGULATION_REFRESH_DAYS)):
            return cached
        items = []
        try:
            feed = feedparser.parse(settings.sebi_rss_url)
            for entry in feed.entries[:limit]:
                items.append(
                    {
                        "title": entry.get("title", "SEBI update"),
                        "summary": entry.get("summary", ""),
                        "published": entry.get("published", datetime.utcnow().isoformat()),
                        "impact": "neutral",
                    }
                )
        except Exception:
            items = []
        if items:
            cache_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
            return items
        return cached or []

    def fetch_policy_updates(self, sector: str, limit: int = 8, force_refresh: bool = False) -> List[Dict]:
        cache_path = settings.policy_cache_dir / f"{sector.replace(' ', '_')}.json"
        cached = self._load_json_cache(cache_path)
        if not force_refresh and cached is not None and self._is_cache_fresh(cache_path, timedelta(days=POLICY_REFRESH_DAYS)):
            return cached
        items = self._download_policy_news(sector, min(limit, 8))
        if items:
            cache_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
            return items
        return cached or []

    def _is_cache_fresh(self, path, max_age: timedelta) -> bool:
        if not path.exists():
            return False
        age = datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)
        return age <= max_age

    def _load_json_cache(self, path) -> List[Dict] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _parse_datetime(self, value: str | None) -> datetime:
        if not value:
            return datetime.min
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except Exception:
            try:
                return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
            except Exception:
                return datetime.min

    def _latest_published(self, articles: List[Dict]) -> datetime:
        dates = [self._parse_datetime(article.get("published")) for article in articles if article.get("published")]
        return max(dates) if dates else datetime.min

    def _merge_articles(self, articles: List[Dict], limit: int) -> List[Dict]:
        seen = set()
        merged = []
        for article in sorted(articles, key=lambda item: self._parse_datetime(item.get("published")), reverse=True):
            key = (article.get("title", "").strip(), article.get("published", "").strip())
            if key in seen:
                continue
            seen.add(key)
            merged.append(article)
            if len(merged) >= limit:
                break
        return merged

    def _load_cached_stock_history(self, cache_path) -> pd.DataFrame | None:
        if not cache_path.exists():
            return None
        try:
            cached = pd.read_csv(cache_path)
        except Exception:
            return None
        if not {"date", "open", "high", "low", "close"}.issubset(set(cached.columns)):
            return None
        source = str(cached.get("data_source", pd.Series(dtype=str)).iloc[0]) if "data_source" in cached.columns and not cached.empty else "unknown"
        if source != "yfinance":
            return None
        return cached

    def _download_from_yfinance(self, ticker: str, years: int) -> pd.DataFrame:
        if yf is None:
            return pd.DataFrame()
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365 * years + 30)
        try:
            frame = yf.download(
                ticker,
                start=start_date.date().isoformat(),
                end=end_date.date().isoformat(),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception:
            return pd.DataFrame()
        if frame.empty:
            return pd.DataFrame()
        frame = self._normalize_yfinance_history(frame)
        usable = frame[[col for col in frame.columns if col in {"date", "open", "high", "low", "close", "volume"}]].copy()
        usable = usable.dropna(subset=["date", "close"])
        if len(usable) < max(252 * min(years, 5), 750):
            return pd.DataFrame()
        usable["ticker"] = ticker
        usable["data_source"] = "yfinance"
        return usable

    def _download_with_alias_support(self, ticker: str, years: int) -> pd.DataFrame:
        attempts = [ticker] + HISTORICAL_TICKER_ALIASES.get(ticker, [])
        for symbol in attempts:
            data = self._download_from_yfinance(symbol, years)
            if not data.empty:
                data["ticker"] = ticker
                if symbol != ticker:
                    data["history_symbol"] = symbol
                return data
        return pd.DataFrame()

    def _normalize_yfinance_history(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        if isinstance(normalized.columns, pd.MultiIndex):
            flattened = []
            for col in normalized.columns:
                parts = [str(part).strip() for part in col if str(part).strip() and str(part).strip().lower() != "nan"]
                flattened.append(parts[0] if parts else "")
            normalized.columns = flattened

        normalized = normalized.reset_index()
        cleaned_columns = []
        for col in normalized.columns:
            label = str(col).strip().lower().replace(" ", "_")
            if label.startswith("adj_close"):
                label = "close"
            elif label.startswith("close"):
                label = "close"
            elif label.startswith("open"):
                label = "open"
            elif label.startswith("high"):
                label = "high"
            elif label.startswith("low"):
                label = "low"
            elif label.startswith("volume"):
                label = "volume"
            elif label in {"datetime", "date_", "index"}:
                label = "date"
            cleaned_columns.append(label)
        normalized.columns = cleaned_columns

        if "date" not in normalized.columns:
            first_col = normalized.columns[0]
            normalized = normalized.rename(columns={first_col: "date"})

        keep_order = ["date", "open", "high", "low", "close", "volume"]
        available = [col for col in keep_order if col in normalized.columns]
        return normalized[available].copy()

    def _download_news(self, company: str, sector: str, limit: int) -> List[Dict]:
        if not settings.news_api_key:
            return []
        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": f'"{company}" OR "{sector}" AND India stock market',
                    "pageSize": limit,
                    "sortBy": "publishedAt",
                    "apiKey": settings.news_api_key,
                    "language": "en",
                },
                timeout=5,
            )
            response.raise_for_status()
            articles = response.json().get("articles", [])
            return [
                {
                    "title": item.get("title", ""),
                    "summary": item.get("description", ""),
                    "source": item.get("source", {}).get("name", "NewsAPI"),
                    "published": item.get("publishedAt", datetime.utcnow().isoformat()),
                }
                for item in articles
            ]
        except Exception:
            return []

    def _download_policy_news(self, sector: str, limit: int) -> List[Dict]:
        if not settings.news_api_key:
            return []
        query = (
            f'("{sector}" OR "india government policy" OR "economic policy" OR "budget" OR "ministry notification") '
            f'AND ("policy" OR "regulation" OR "reform" OR "subsidy" OR "incentive" OR "government")'
        )
        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "pageSize": limit,
                    "sortBy": "publishedAt",
                    "apiKey": settings.news_api_key,
                    "language": "en",
                },
                timeout=5,
            )
            response.raise_for_status()
            articles = response.json().get("articles", [])
            records = []
            for item in articles:
                text = f"{item.get('title', '')} {item.get('description', '')}".lower()
                impact = "positive"
                if any(token in text for token in ["ban", "penalty", "delay", "restriction", "compliance burden"]):
                    impact = "negative"
                elif any(token in text for token in ["support", "approval", "incentive", "investment", "allocation", "capex"]):
                    impact = "positive"
                else:
                    impact = "neutral"
                records.append(
                    {
                        "title": item.get("title", ""),
                        "summary": item.get("description", ""),
                        "source": item.get("source", {}).get("name", "NewsAPI"),
                        "published": item.get("publishedAt", datetime.utcnow().isoformat()),
                        "impact": impact,
                    }
                )
            return records
        except Exception:
            return []

    def _download_macro_news(self, sector: str, limit: int) -> List[Dict]:
        if not settings.news_api_key:
            return []
        query = f'("{sector}" OR "india economy" OR "global markets" OR "crude oil" OR "interest rates" OR "inflation") AND ("stocks" OR "market" OR "policy")'
        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "pageSize": limit,
                    "sortBy": "publishedAt",
                    "apiKey": settings.news_api_key,
                    "language": "en",
                },
                timeout=5,
            )
            response.raise_for_status()
            articles = response.json().get("articles", [])
            return [
                {
                    "title": item.get("title", ""),
                    "summary": item.get("description", ""),
                    "source": item.get("source", {}).get("name", "NewsAPI"),
                    "published": item.get("publishedAt", datetime.utcnow().isoformat()),
                }
                for item in articles
            ]
        except Exception:
            return []
