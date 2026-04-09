from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.config import ensure_directories, settings
from backend.schemas import BacktestRequest, BacktestResponse, CompanyRequest, CompanyResponse, PortfolioRequest, PortfolioResponse, SectorRequest, SectorResponse
from backend.service import FinancialAdvisorService

app = FastAPI(title="AI Financial Advisory System", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
service = FinancialAdvisorService()


@app.on_event("startup")
def startup_event() -> None:
    ensure_directories()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/sectors")
def sectors() -> dict:
    return {"sectors": service.list_sectors()}


@app.post("/portfolio", response_model=PortfolioResponse)
def portfolio(request: PortfolioRequest) -> dict:
    try:
        return service.portfolio_recommendation(
            request.budget,
            request.risk_tolerance,
            request.holding_period,
            request.selected_sectors,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sector", response_model=SectorResponse)
def sector(request: SectorRequest) -> dict:
    try:
        return service.sector_analysis(request.sector)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/company", response_model=CompanyResponse)
def company(request: CompanyRequest) -> dict:
    try:
        return service.company_analysis(request.company_keyword, request.total_budget, request.days)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/backtest", response_model=BacktestResponse)
def backtest(request: BacktestRequest) -> dict:
    try:
        return service.backtest_company(request.company_keyword, request.days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/download/{filename}")
def download(filename: str) -> FileResponse:
    path = settings.outputs_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=Path(path), filename=filename, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
