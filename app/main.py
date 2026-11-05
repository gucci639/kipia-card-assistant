"""FastAPI веб-сервис для генерации B2B-карточек контрольно-измерительных приборов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.feed_importer import fetch_offer_from_yml_feed, fetch_product_from_url
from app.llm_service import LLMServiceError, ProductCardGenerator
from app.schemas import ProductCardDraft, ProductInput
from app.validators import analyze_input_completeness

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
INDEX_HTML_PATH = BASE_DIR / "static" / "index.html"

app = FastAPI(
    title="AI-помощник для подготовки товарных описаний КИПиА",
    description="Прототип сервиса генерации карточек приборов для B2B-интернет-магазина",
    version="1.0.0",
)


@app.exception_handler(LLMServiceError)
async def llm_error_handler(_request: Request, exc: LLMServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "type": "LLMServiceError"},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "Передан некорректный входной JSON или нарушена схема данных.",
            "details": exc.errors(),
            "type": "ValidationError",
        },
    )


class FeedLookupRequest(BaseModel):
    query: str = Field(..., min_length=1, description="ID предложения, артикул, название в YML-фиде или прямой URL товара")
    enrich_from_web: bool = Field(default=True, description="Дообогащать ли карточку с HTML-страницы товара")


@app.get("/", response_class=HTMLResponse)
async def index_page() -> HTMLResponse:
    if not INDEX_HTML_PATH.exists():
        return HTMLResponse("<h1>UI не найден</h1>", status_code=404)
    return HTMLResponse(INDEX_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    gen = ProductCardGenerator()
    has_key = bool(gen.settings.resolved_api_key())
    return {
        "status": "ok",
        "llm_configured": has_key,
        "model": gen.settings.llm_model,
        "base_url": gen.settings.llm_base_url,
    }


@app.get("/api/samples")
async def get_samples() -> list[dict[str, Any]]:
    sample_file = DATA_DIR / "sample_inputs.json"
    if not sample_file.exists():
        return []
    return json.loads(sample_file.read_text(encoding="utf-8"))


@app.post("/api/validate")
async def validate_input_only(product: ProductInput) -> dict[str, Any]:
    """Быстрая проверка полноты входных данных без расхода токенов LLM."""
    report = analyze_input_completeness(product)
    return report.model_dump()


@app.post("/api/import-yml", response_model=ProductInput)
async def import_from_yml_or_url(req: FeedLookupRequest) -> ProductInput:
    """Импортирует данные о товаре из YML-фида Союз-Прибор (по ID/артикулу/названию) или напрямую по URL."""
    q = req.query.strip()
    try:
        if q.startswith("http://") or q.startswith("https://"):
            return await fetch_product_from_url(q)
        return await fetch_offer_from_yml_feed(q, enrich_from_web=req.enrich_from_web)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка при загрузке данных из источника: {exc}",
        ) from exc


@app.post("/api/generate", response_model=ProductCardDraft)
async def generate_product_card(product: ProductInput) -> ProductCardDraft:
    """Формирует черновик B2B-карточки товара через LLM с верификацией характеристик."""
    generator = ProductCardGenerator()
    return await generator.generate_card(product)
