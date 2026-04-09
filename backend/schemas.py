from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PortfolioRequest(BaseModel):
    budget: float = Field(..., gt=0)
    risk_tolerance: str
    holding_period: str
    selected_sectors: List[str] = Field(default_factory=list)


class SectorRequest(BaseModel):
    sector: str


class CompanyRequest(BaseModel):
    company_keyword: str
    total_budget: Optional[float] = Field(default=None, gt=0)
    days: int = Field(default=30, ge=15, le=90)


class BacktestRequest(BaseModel):
    company_keyword: str
    days: int = Field(default=7, ge=7, le=90)


class DownloadResponse(BaseModel):
    filename: str
    absolute_path: str


class RecommendationItem(BaseModel):
    company: str
    ticker: str
    sector: str
    current_price: float
    predicted_price: float
    growth_pct: float
    growth_probability: float
    allocation_pct: float
    allocation_amount: float
    holding_time_days: int
    risk_level: str
    explanation: str
    short_term_signal_pct: Optional[float] = None
    recommendation: Optional[str] = None
    prediction_horizon_days: Optional[int] = None
    trend_90d: Optional[str] = None
    growth_90d: Optional[float] = None


class PortfolioResponse(BaseModel):
    portfolio: List[RecommendationItem]
    excel_file: DownloadResponse
    summary: dict


class SectorResponse(BaseModel):
    sector: str
    recommendations: List[RecommendationItem]
    excel_file: DownloadResponse
    summary: dict


class CompanyResponse(BaseModel):
    company: dict
    insights: dict
    chart: dict
    excel_file: DownloadResponse


class BacktestResponse(BaseModel):
    company: dict
    summary: dict
    results: list[dict]
    chart: list[dict]
