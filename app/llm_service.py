"""Сервис обращения к LLM API с обработкой ошибок, парсингом JSON и контролем ограничений."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from app.config import Settings, get_settings
from app.prompts import (
    SYSTEM_PROMPT,
    build_length_correction_prompt,
    build_user_prompt,
)
from app.schemas import (
    CommercialBlock,
    ProductCardDraft,
    ProductInput,
    SpecItem,
)
from app.validators import (
    analyze_input_completeness,
    detect_unverified_numbers,
    enforce_seo_title_limit,
    merge_and_verify_specs,
    render_exports,
)

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Пользовательское исключение при ошибках конфигурации или обращения к LLM API."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Извлекает JSON-объект из ответа модели (даже если модель обернула его в ```json ... ```)."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace : last_brace + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMServiceError(
            f"Модель вернула некорректный JSON: {exc.msg} (позиция {exc.pos}).",
            status_code=502,
        ) from exc

    if not isinstance(parsed, dict):
        raise LLMServiceError("Ответ модели не является JSON-объектом.", status_code=502)
    return parsed


class ProductCardGenerator:
    """Генератор карточек товаров КИПиА с многоуровневой защитой от галлюцинаций."""

    def __init__(self, settings: Settings | None = None, client: AsyncOpenAI | None = None):
        self.settings = settings or get_settings()
        self._client = client

    def _get_client(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client
        api_key = self.settings.resolved_api_key()
        if not api_key:
            raise LLMServiceError(
                "Не задан API-ключ LLM. Укажите переменную окружения LLM_API_KEY в файле .env (см. .env.example).",
                status_code=500,
            )
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.settings.llm_base_url,
            timeout=self.settings.llm_timeout_seconds,
        )
        return self._client

    async def _call_llm_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Выполняет запрос к LLM API и возвращает распарсенный JSON-словарь."""
        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.settings.llm_temperature,
                response_format={"type": "json_object"},
            )
        except AuthenticationError as exc:
            raise LLMServiceError(
                "Ошибка авторизации в LLM API (401). Проверьте корректность LLM_API_KEY в .env.",
                status_code=401,
            ) from exc
        except RateLimitError as exc:
            raise LLMServiceError(
                "Превышен лимит запросов к LLM API (429 Rate Limit). Повторите попытку через несколько секунд.",
                status_code=429,
            ) from exc
        except APITimeoutError as exc:
            raise LLMServiceError(
                f"Превышено время ожидания ответа от LLM API ({self.settings.llm_timeout_seconds} сек).",
                status_code=504,
            ) from exc
        except APIConnectionError as exc:
            raise LLMServiceError(
                f"Не удалось подключиться к LLM API ({self.settings.llm_base_url}). Проверьте интернет-соединение и базовый URL.",
                status_code=502,
            ) from exc
        except APIStatusError as exc:
            raise LLMServiceError(
                f"Ошибка LLM API (HTTP {exc.status_code}): {exc.message}",
                status_code=502,
            ) from exc
        except APIError as exc:
            raise LLMServiceError(
                f"Ошибка при обращении к LLM API: {exc}",
                status_code=502,
            ) from exc

        if not response.choices or not response.choices[0].message.content:
            raise LLMServiceError("LLM API вернул пустой ответ.", status_code=502)

        return extract_json_object(response.choices[0].message.content)

    async def generate_card(self, product: ProductInput) -> ProductCardDraft:
        """Основной пайплайн генерации черновика карточки товара по входным данным."""
        # Шаг 0: проверяем, что входной объект не является полностью пустым
        if not product.name and not product.manufacturer and not product.specs and not product.description:
            raise LLMServiceError(
                "Входные данные полностью пусты: укажите хотя бы название прибора или технические характеристики.",
                status_code=422,
            )

        # Шаг 1: детерминированный аудит полноты данных
        completeness = analyze_input_completeness(product)

        # Шаг 2: основной запрос к LLM
        user_prompt = build_user_prompt(product, completeness.missing_fields)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw_data = await self._call_llm_json(messages)

        seo_title = str(raw_data.get("seo_title") or product.name or "Прибор КИПиА").strip()
        seo_title = enforce_seo_title_limit(seo_title, max_len=70)

        seo_description = re.sub(r"\s+", " ", str(raw_data.get("seo_description") or "")).strip()
        unverified_nums = detect_unverified_numbers(product, seo_description)

        # Шаг 3: авто-коррекция длины seo_description (800..1200 символов) и галлюцинаций цифр (до 2 итераций)
        attempts = 0
        while (not (800 <= len(seo_description) <= 1200) or unverified_nums) and attempts < 2:
            attempts += 1
            logger.info(
                "Коррекция seo_description (попытка %d): текущая длина=%d, непроверенные числа=%s",
                attempts,
                len(seo_description),
                unverified_nums,
            )
            correction_prompt = build_length_correction_prompt(
                product=product,
                current_description=seo_description,
                unverified_nums=unverified_nums,
            )
            fixed_data = await self._call_llm_json(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": correction_prompt},
                ]
            )
            candidate_desc = re.sub(r"\s+", " ", str(fixed_data.get("seo_description") or "")).strip()
            if candidate_desc:
                seo_description = candidate_desc
                unverified_nums = detect_unverified_numbers(product, seo_description)

        # Страховочный зажим длины на случай отклонения после ретраев
        if len(seo_description) > 1200:
            cut = seo_description[:1195]
            last_dot = max(cut.rfind(". "), cut.rfind("."), cut.rfind("; "))
            if last_dot >= 800:
                seo_description = cut[: last_dot + 1].strip()
            else:
                seo_description = cut.rstrip() + "."

        # Шаг 4: объединение и верификация характеристик (никаких выдуманных значений)
        raw_specs = raw_data.get("structured_specs") or []
        parsed_llm_specs: list[SpecItem] = []
        if isinstance(raw_specs, list):
            for item in raw_specs:
                if isinstance(item, dict) and item.get("name"):
                    val = str(item.get("value") or "данных нет").strip()
                    st = "missing" if val.lower() in ("данных нет", "нет данных", "н/д", "-") else "present"
                    parsed_llm_specs.append(SpecItem(name=str(item["name"]).strip(), value=val, status=st))

        verified_specs = merge_and_verify_specs(product, parsed_llm_specs)

        # Шаг 5: поисковые запросы (3..5 штук)
        raw_queries = raw_data.get("search_queries") or []
        queries = [str(q).strip() for q in raw_queries if str(q).strip()][:5]
        base_label = f"{product.name} {product.manufacturer}".strip() or "прибор КИПиА"
        fallback_queries = [
            f"{base_label} купить",
            f"{base_label} цена и характеристики",
            f"{product.name or 'прибор КИПиА'} техпаспорт и поверка",
        ]
        for fq in fallback_queries:
            if len(queries) < 3 and fq not in queries:
                queries.append(fq)

        # Шаг 6: commercial-блок
        raw_comm = raw_data.get("commercial_block") or {}
        if not isinstance(raw_comm, dict):
            raw_comm = {}
        comm_block = CommercialBlock(
            target_audience=[str(x).strip() for x in (raw_comm.get("target_audience") or []) if str(x).strip()]
            or ["Отделы комплектации и снабжения промышленных предприятий", "Инженеры КИПиА и АСУ ТП"],
            solved_tasks=[str(x).strip() for x in (raw_comm.get("solved_tasks") or []) if str(x).strip()]
            or ["Контроль технологических параметров в промышленных контурах"],
            key_advantages=[str(x).strip() for x in (raw_comm.get("key_advantages") or []) if str(x).strip()]
            or ["Промышленное исполнение согласно технической документации производителя"],
        )

        # Шаг 7: дополняем вопросы для контент-менеджера рекомендациями LLM
        llm_questions = [
            str(q).strip()
            for q in (raw_data.get("clarification_questions") or [])
            if str(q).strip()
        ]
        for q in llm_questions:
            if q not in completeness.clarification_questions and len(completeness.clarification_questions) < 8:
                completeness.clarification_questions.append(q)

        completeness.unverified_numbers_flagged = unverified_nums

        draft = ProductCardDraft(
            seo_title=seo_title,
            seo_title_length=len(seo_title),
            seo_description=seo_description,
            seo_description_length=len(seo_description),
            structured_specs=verified_specs,
            search_queries=queries,
            commercial_block=comm_block,
            completeness=completeness,
        )
        md_exp, txt_exp = render_exports(draft)
        draft.markdown_export = md_exp
        draft.text_export = txt_exp
        return draft
