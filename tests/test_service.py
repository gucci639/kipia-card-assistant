"""Автотесты для проверки схем, анти-галлюцинационного фильтра, обработки ошибок API и граничных кейсов."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.feed_importer import parse_souz_pribor_html
from app.llm_service import LLMServiceError, ProductCardGenerator, extract_json_object
from app.main import app
from app.schemas import ProductInput, SpecItem
from app.validators import (
    analyze_input_completeness,
    detect_unverified_numbers,
    enforce_seo_title_limit,
    merge_and_verify_specs,
)


def test_product_input_normalization() -> None:
    """Пустые значения в specs должны автоматически заменяться на 'данных нет'."""
    prod = ProductInput(
        name="  Манометр показывающий DM 02  ",
        manufacturer=" Манотомь ",
        specs={"диапазон измерений": "0–1,6 МПа", "питание": "", "интерфейсы": None},  # type: ignore[dict-item]
        description="  Описание прибора  ",
    )
    assert prod.name == "Манометр показывающий DM 02"
    assert prod.manufacturer == "Манотомь"
    assert prod.specs["диапазон измерений"] == "0–1,6 МПа"
    assert prod.specs["питание"] == "данных нет"
    assert prod.specs["интерфейсы"] == "данных нет"


def test_completeness_report_sparse_data() -> None:
    """При неполных данных (например, голый оффер из YML без описания) анализатор фиксирует недостающие поля."""
    sparse = ProductInput(
        name="Gsm-модем ПМ01-24.АВ",
        manufacturer="ОВЕН",
        specs={"Артикул": "39432", "Код товара": "73534"},
        description="",
    )
    report = analyze_input_completeness(sparse)
    assert report.is_sufficient is False
    assert report.completeness_score < 70
    assert any("description" in f for f in report.missing_fields)
    assert len(report.clarification_questions) >= 3


def test_anti_hallucination_spec_merger() -> None:
    """Если LLM пытается подставить выдуманную характеристику с цифрой, которой нет во входе, она отбрасывается."""
    prod = ProductInput(
        name="Манометр DM 02",
        manufacturer="Манотомь",
        specs={"диапазон измерений": "0–1,6 МПа", "класс точности": "1,5"},
        description="Корпус диаметром 100 мм.",
    )
    llm_specs = [
        SpecItem(name="Диаметр корпуса", value="100 мм", status="present"),  # Есть в description -> ок
        SpecItem(name="Потребляемая мощность", value="45 Вт", status="present"),  # Галлюцинация (45 нет во входе) -> должна быть отброшена
        SpecItem(name="Степень защиты", value="данных нет", status="missing"),  # Честное 'данных нет' -> сохраняется
    ]
    merged = merge_and_verify_specs(prod, llm_specs)
    names_to_vals = {s.name: s.value for s in merged}

    assert names_to_vals["диапазон измерений"] == "0–1,6 МПа"
    assert names_to_vals["Диаметр корпуса"] == "100 мм"
    assert "Потребляемая мощность" not in names_to_vals
    assert names_to_vals["Степень защиты"] == "данных нет"
    # Обязательная группа питания должна быть добавлена со значением "данных нет"
    assert any("Питание" in s.name and s.value == "данных нет" for s in merged)


def test_detect_unverified_numbers() -> None:
    """Детектор чисел должен ловить вымышленные технические номиналы в тексте описания."""
    prod = ProductInput(
        name="Манометр DM 02",
        manufacturer="Манотомь",
        specs={"диапазон измерений": "0–1,6 МПа", "класс точности": "1,5", "диаметр корпуса": "100 мм"},
        description="Температура среды до 150 °C.",
    )
    clean_text = "Манометр DM 02 с корпусом 100 мм и классом точности 1,5 измеряет давление до 1,6 МПа при 150 °C."
    assert detect_unverified_numbers(prod, clean_text) == []

    hallucinated_text = clean_text + " Степень защиты IP68, гарантия 5 лет, питание 220 В."
    flagged = detect_unverified_numbers(prod, hallucinated_text)
    assert "68" in flagged or "220" in flagged or "5" in flagged


def test_enforce_seo_title_limit() -> None:
    long_title = "Манометр показывающий технический МП3-У Манотомь 0–1 кгс/см² класс точности 1.5 корпус 100 мм"
    shortened = enforce_seo_title_limit(long_title, max_len=70)
    assert len(shortened) <= 70


def test_extract_json_object_handles_markdown_fences() -> None:
    wrapped = 'Вот результат:\n```json\n{"seo_title": "Тест", "seo_description": "Текст"}\n```'
    data = extract_json_object(wrapped)
    assert data["seo_title"] == "Тест"


@pytest.mark.asyncio
async def test_generator_empty_input_raises_422() -> None:
    gen = ProductCardGenerator(settings=Settings(llm_api_key="test-key"))
    empty_prod = ProductInput()
    with pytest.raises(LLMServiceError) as exc_info:
        await gen.generate_card(empty_prod)
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_generator_missing_api_key_raises_500(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "XKIRO_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    gen = ProductCardGenerator(settings=Settings(llm_api_key=""))
    prod = ProductInput(name="Манометр DM 02", manufacturer="Манотомь")
    with pytest.raises(LLMServiceError) as exc_info:
        await gen.generate_card(prod)
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_generator_full_pipeline_with_mock_llm() -> None:
    """Проверяем полный цикл генерации карточки с моком OpenAI клиента."""
    desc_900_chars = (
        "Манометр показывающий DM 02 производства компании Манотомь предназначен для непрерывного измерения "
        "избыточного давления неагрессивных к медным сплавам жидких и газообразных сред, не являющихся вязкими "
        "или кристаллизующимися. Прибор рассчитан на эксплуатацию при рабочей температуре измеряемой среды до 150 °C "
        "и широко востребован на объектах теплоснабжения, водоснабжения, в системах ЖКХ и промышленных контурах. "
        "Модель выполнена в прочном стальном корпусе диаметром 100 мм с измерительным механизмом из латунного сплава. "
        "Диапазон измерений составляет 0–1,6 МПа при классе точности 1,5, что обеспечивает достоверный визуальный "
        "контроль технологического процесса. Подключение прибора к трубопроводу осуществляется через стандартный "
        "резьбовой штуцер G1/2. Конструкция не требует внешнего источника электропитания. Сведения о степени защиты IP "
        "и межповерочном интервале во входных данных отсутствуют и уточняются по паспорту изделия при оформлении заказа."
    )
    assert 800 <= len(desc_900_chars) <= 1200

    mock_payload: dict[str, Any] = {
        "seo_title": "Манометр показывающий DM 02 Манотомь (0–1,6 МПа, кл. 1,5)",
        "seo_description": desc_900_chars,
        "structured_specs": [
            {"name": "Диапазон измерений", "value": "0–1,6 МПа", "status": "present"},
            {"name": "Класс точности", "value": "1,5", "status": "present"},
            {"name": "Питание", "value": "данных нет", "status": "missing"},
        ],
        "search_queries": [
            "манометр dm 02 манотомь купить",
            "манометр 0-1.6 мпа g1/2 100 мм",
            "манометр показывающий dm 02 цена",
        ],
        "commercial_block": {
            "target_audience": ["Службы КИПиА", "Отделы снабжения ЖКХ"],
            "solved_tasks": ["Контроль давления в сетях теплоснабжения"],
            "key_advantages": ["Стальной корпус 100 мм и штуцер G1/2"],
        },
        "clarification_questions": ["Уточните степень защиты IP по паспорту прибора."],
    }

    mock_msg = MagicMock()
    mock_msg.content = json.dumps(mock_payload, ensure_ascii=False)
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    gen = ProductCardGenerator(settings=Settings(llm_api_key="mock-key"), client=mock_client)
    prod = ProductInput(
        name="Манометр показывающий DM 02",
        manufacturer="Манотомь",
        specs={
            "диапазон измерений": "0–1,6 МПа",
            "класс точности": "1,5",
            "присоединение": "G1/2",
            "диаметр корпуса": "100 мм",
        },
        description="Прибор предназначен для измерения избыточного давления сред с температурой до 150 °C.",
    )
    draft = await gen.generate_card(prod)
    assert draft.seo_title_length <= 70
    assert 800 <= draft.seo_description_length <= 1200
    assert 3 <= len(draft.search_queries) <= 5
    assert draft.completeness.unverified_numbers_flagged == []
    assert "Манометр показывающий DM 02" in draft.markdown_export


def test_fastapi_invalid_json_returns_422() -> None:
    client = TestClient(app)
    resp = client.post(
        "/api/generate",
        content="{broken json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert "error" in body


def test_parse_souz_pribor_html_extracts_fields() -> None:
    sample_html = """
    <html><body>
      <h1>МП3-У - манометр показывающий</h1>
      <div>Манотомь (Россия)</div>
      <div class="characteristics__item"><div>Диаметр корпуса:</div><div>100 мм</div></div>
      <div class="description-specifications__content-inner">
        Манометр МП3-У предназначен для измерения избыточного давления.
      </div>
      <div class="description-specifications__content-inner" id="specifications">
        <table><tr><td>Класс точности</td><td>1.5</td></tr></table>
      </div>
    </body></html>
    """
    parsed = parse_souz_pribor_html(sample_html, fallback_url="https://www.souz-pribor.ru/product/mp3-u/")
    assert parsed.name == "МП3-У - манометр показывающий"
    assert parsed.manufacturer == "Манотомь"
    assert parsed.specs["Диаметр корпуса"] == "100 мм"
    assert parsed.specs["Класс точности"] == "1.5"
    assert "избыточного давления" in parsed.description
