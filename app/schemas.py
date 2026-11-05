"""Pydantic-схемы входных данных, ответа сервиса и структурированного результата."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class ProductInput(BaseModel):
    """Входные данные о контрольно-измерительном приборе."""

    name: str = Field(
        default="",
        description="Название товара (например, Манометр показывающий МП3-У)",
    )
    manufacturer: str = Field(
        default="",
        description="Производитель прибора (например, Манотомь, ОВЕН)",
    )
    specs: dict[str, str] = Field(
        default_factory=dict,
        description="Словарь технических характеристик (ключ — название параметра, значение — величина)",
    )
    description: str = Field(
        default="",
        description="Текстовое описание или фрагмент технического паспорта прибора",
    )
    source_url: str | None = Field(
        default=None,
        description="Опциональная ссылка на источник (карточку в каталоге или YML-фид)",
    )

    @field_validator("name", "manufacturer", "description", mode="before")
    @classmethod
    def clean_string(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("specs", mode="before")
    @classmethod
    def normalize_specs(cls, v: Any) -> dict[str, str]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("Поле 'specs' должно быть JSON-объектом (словарём характеристик).")
        cleaned: dict[str, str] = {}
        for k, val in v.items():
            key_str = str(k).strip()
            if not key_str:
                continue
            val_str = "" if val is None else str(val).strip()
            cleaned[key_str] = val_str if val_str else "данных нет"
        return cleaned


class SpecItem(BaseModel):
    """Отдельная строка в структурированном списке характеристик."""

    name: str = Field(..., description="Наименование характеристики")
    value: str = Field(
        ...,
        description="Значение характеристики строго из входных данных или 'данных нет'",
    )
    status: Literal["present", "missing"] = Field(
        default="present",
        description="Статус заполненности: present — есть во входных данных, missing — данных нет",
    )


class CommercialBlock(BaseModel):
    """Краткий commercial-блок для контент-менеджера и отдела продаж B2B."""

    target_audience: list[str] = Field(
        default_factory=list,
        description="Кому подходит прибор (целевые сегменты B2B, роли закупщиков и инженеров)",
    )
    solved_tasks: list[str] = Field(
        default_factory=list,
        description="Какие инженерные и производственные задачи решает прибор",
    )
    key_advantages: list[str] = Field(
        default_factory=list,
        description="Аргументы для КП и общения с заказчиком (только на основе фактов из входных данных)",
    )


class CompletenessReport(BaseModel):
    """Отчёт о полноте входных данных и проверке отсутствия выдуманных фактов."""

    is_sufficient: bool = Field(
        ...,
        description="Достаточно ли входных данных для полноценной публикации без уточнений",
    )
    completeness_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Оценка полноты входных данных от 0 до 100%",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Список отсутствующих или пустых базовых и технических полей",
    )
    clarification_questions: list[str] = Field(
        default_factory=list,
        description="Что контент-менеджеру необходимо уточнить у поставщика или в техпаспорте",
    )
    unverified_numbers_flagged: list[str] = Field(
        default_factory=list,
        description="Числа/номиналы в сгенерированном тексте, которых не было во входных данных (анти-галлюцинационный фильтр)",
    )


class ProductCardDraft(BaseModel):
    """Итоговый черновик карточки товара для B2B-интернет-магазина."""

    seo_title: str = Field(
        ...,
        description="SEO-заголовок страницы товара (строго до 70 символов)",
    )
    seo_title_length: int = Field(
        ...,
        description="Фактическая длина SEO-заголовка в символах",
    )
    seo_description: str = Field(
        ...,
        description="SEO-описание товара объёмом 800–1200 символов",
    )
    seo_description_length: int = Field(
        ...,
        description="Фактическая длина SEO-описания в символах",
    )
    structured_specs: list[SpecItem] = Field(
        ...,
        description="Структурированный список ключевых характеристик (с пометками 'данных нет' для отсутствующих)",
    )
    search_queries: list[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3–5 целевых поисковых запросов для SEO-оптимизации карточки",
    )
    commercial_block: CommercialBlock = Field(
        ...,
        description="Краткий commercial-блок для менеджера: кому подходит прибор и какие задачи решает",
    )
    completeness: CompletenessReport = Field(
        ...,
        description="Отчёт о недостающих полях, требующих уточнения, и проверке фактов",
    )
    markdown_export: str = Field(
        default="",
        description="Готовое представление карточки в формате Markdown для копирования",
    )
    text_export: str = Field(
        default="",
        description="Готовое текстовое представление карточки для быстрой вставки в CMS",
    )
