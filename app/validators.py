"""Детерминированная валидация полноты данных, защита от галлюцинаций и форматирование экспортов."""

from __future__ import annotations

import re
from app.schemas import (
    CompletenessReport,
    ProductCardDraft,
    ProductInput,
    SpecItem,
)

# Ожидаемые ключевые характеристики для приборов КИПиА (согласно п. 2 ТЗ):
# диапазон измерения, класс точности, питание, тип корпуса/защита, интерфейсы/присоединение
EXPECTED_SPEC_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    (
        "Диапазон измерений / входной сигнал",
        ("диапазон", "измерен", "предел", "давлен", "температур", "вход", "канал"),
    ),
    (
        "Класс точности / погрешность",
        ("точн", "погрешн", "кл.т", "класс"),
    ),
    (
        "Питание / энергопотребление",
        ("питан", "напряжен", "мощност", "автономн", "ток"),
    ),
    (
        "Тип корпуса / габариты / степень защиты",
        ("корпус", "диаметр", "габарит", "защит", "ip", "масса", "монтаж"),
    ),
    (
        "Интерфейсы / присоединение / выходы",
        ("интерфейс", "протокол", "присоедин", "резьб", "штуцер", "выход", "rs-485", "modbus"),
    ),
]


def analyze_input_completeness(product: ProductInput) -> CompletenessReport:
    """Проверяет входные данные на наличие обязательных полей и ключевых групп характеристик."""
    missing_fields: list[str] = []
    clarifications: list[str] = []

    if not product.name:
        missing_fields.append("name (Название товара)")
        clarifications.append("Укажите точное наименование и модификацию прибора.")

    if not product.manufacturer:
        missing_fields.append("manufacturer (Производитель)")
        clarifications.append("Уточните завод-изготовитель или бренд прибора.")

    if not product.description or len(product.description) < 30:
        missing_fields.append("description (Текстовое описание / фрагмент техпаспорта)")
        clarifications.append(
            "Текстовое описание отсутствует или слишком короткое: рекомендуется добавить выдержку из паспорта/руководства по эксплуатации (назначение, рабочая среда, особенности монтажа)."
        )

    valid_specs = {
        k: v
        for k, v in product.specs.items()
        if v and v.strip().lower() not in ("", "данных нет", "н/д", "нет данных", "-", "none", "null")
    }

    if not valid_specs:
        missing_fields.append("specs (Технические характеристики)")
        clarifications.append("Словарь характеристик пуст: необходимо заполнить технические параметры прибора.")

    # Проверяем покрытие 5 ключевых инженерных групп КИПиА (из specs + description + name)
    combined_source_lower = " ".join(
        [
            product.name.lower(),
            product.description.lower(),
            " ".join(f"{k} {v}".lower() for k, v in valid_specs.items()),
        ]
    )

    specs_keys_lower = " ".join(k.lower() for k in valid_specs.keys())

    for group_label, keywords in EXPECTED_SPEC_GROUPS:
        in_specs = any(kw in specs_keys_lower for kw in keywords)
        in_text = any(kw in combined_source_lower for kw in keywords)
        if not in_specs and not in_text:
            missing_fields.append(f"Характеристика: {group_label}")
            clarifications.append(
                f"В данных не указан параметр «{group_label}» — уточните по техпаспорту (если применимо к данному типу прибора)."
            )

    # Расчёт скора полноты (0..100)
    score = 100
    if not product.name:
        score -= 30
    if not product.manufacturer:
        score -= 15
    if not product.description or len(product.description) < 30:
        score -= 20
    if len(valid_specs) == 0:
        score -= 25
    elif len(valid_specs) < 3:
        score -= 15

    missing_spec_groups = [m for m in missing_fields if m.startswith("Характеристика:")]
    score -= min(25, len(missing_spec_groups) * 5)
    score = max(0, min(100, score))

    is_sufficient = (
        bool(product.name)
        and bool(product.manufacturer)
        and len(valid_specs) >= 3
        and len(product.description) >= 30
    )

    return CompletenessReport(
        is_sufficient=is_sufficient,
        completeness_score=score,
        missing_fields=missing_fields,
        clarification_questions=clarifications,
        unverified_numbers_flagged=[],
    )


def merge_and_verify_specs(
    product: ProductInput,
    llm_specs: list[SpecItem],
) -> list[SpecItem]:
    """Гарантирует, что все входные характеристики сохранены без искажений,
    а отсутствующие базовые категории явно помечены как 'данных нет'."""
    result: list[SpecItem] = []
    seen_keys: set[str] = set()

    # 1. Сначала берём все характеристики напрямую из входного JSON (источник истины)
    for raw_k, raw_v in product.specs.items():
        k_clean = raw_k.strip()
        v_clean = raw_v.strip()
        is_missing = v_clean.lower() in ("", "данных нет", "н/д", "нет данных", "-", "none")
        result.append(
            SpecItem(
                name=k_clean,
                value="данных нет" if is_missing else v_clean,
                status="missing" if is_missing else "present",
            )
        )
        seen_keys.add(k_clean.lower())

    # 2. Добавляем извлечённые моделью характеристики из текста паспорта,
    # только если их значения реально присутствуют во входном тексте description/name
    source_blob = f"{product.name} {product.manufacturer} {product.description}".lower()
    for item in llm_specs:
        k_low = item.name.strip().lower()
        if not k_low or k_low in seen_keys:
            continue
        val_clean = item.value.strip()
        if val_clean.lower() in ("данных нет", "нет данных", "н/д", "не указано", "-"):
            result.append(SpecItem(name=item.name.strip(), value="данных нет", status="missing"))
            seen_keys.add(k_low)
        else:
            # Проверяем, что все числа из значения реально есть во входных данных
            nums = re.findall(r"\d+(?:[.,]\d+)?", val_clean)
            all_nums_present = all(
                n in source_blob or n.replace(",", ".") in source_blob or n.replace(".", ",") in source_blob
                for n in nums
            )
            if all_nums_present and (nums or val_clean.lower() in source_blob):
                result.append(SpecItem(name=item.name.strip(), value=val_clean, status="present"))
                seen_keys.add(k_low)

    # 3. Проверяем 5 базовых групп из ТЗ: если группа вообще не представлена — добавляем строку "данных нет"
    all_keys_text = " ".join(s.name.lower() for s in result)
    for group_label, keywords in EXPECTED_SPEC_GROUPS:
        if not any(kw in all_keys_text for kw in keywords):
            result.append(
                SpecItem(
                    name=group_label,
                    value="данных нет",
                    status="missing",
                )
            )

    return result


def extract_numbers(text: str) -> set[str]:
    """Извлекает числовые токены (нормализуя запятую в точку) для сверки фактов."""
    raw = re.findall(r"\b\d+(?:[.,]\d+)?\b", text)
    return {r.replace(",", ".") for r in raw}


def detect_unverified_numbers(product: ProductInput, generated_text: str) -> list[str]:
    """Находит числовые значения в сгенерированном тексте, которых не было во входных данных."""
    source_text = " ".join(
        [
            product.name,
            product.manufacturer,
            product.description,
            " ".join(f"{k} {v}" for k, v in product.specs.items()),
        ]
    )
    source_nums = extract_numbers(source_text)
    # Разрешаем тривиальные числительные (1, 2, 3, 4, 5) и B2B/24, если они в составе 'B2B'
    cleaned_gen = re.sub(r"\bB2B\b", "", generated_text, flags=re.IGNORECASE)
    gen_nums = extract_numbers(cleaned_gen)

    allowed_trivial = {"1", "2", "3"}
    unverified = sorted(gen_nums - source_nums - allowed_trivial)
    return unverified


def enforce_seo_title_limit(title: str, max_len: int = 70) -> str:
    """Гарантирует длину SEO-заголовка строго <= 70 символов без обрыва слова на полуслове."""
    cleaned = re.sub(r"\s+", " ", title).strip()
    if len(cleaned) <= max_len:
        return cleaned
    # Пробуем обрезать по разделителям (' — ', ' | ', ', ', ' ')
    for sep in (" — ", " | ", ": ", ", ", " "):
        cut = cleaned[:max_len]
        idx = cut.rfind(sep)
        if idx >= 35:
            return cut[:idx].rstrip(" ,;:-—|")
    return cleaned[: max_len - 1].rstrip() + "…"


def render_exports(draft: ProductCardDraft) -> tuple[str, str]:
    """Формирует удобные для копирования Markdown и Plain-Text представления карточки."""
    specs_md_lines = [
        f"| {s.name} | {s.value} |" for s in draft.structured_specs
    ]
    specs_text_lines = [
        f"• {s.name}: {s.value}" for s in draft.structured_specs
    ]

    queries_md = "\n".join(f"- `{q}`" for q in draft.search_queries)
    queries_txt = "\n".join(f"• {q}" for q in draft.search_queries)

    aud_md = "\n".join(f"- {a}" for a in draft.commercial_block.target_audience)
    tasks_md = "\n".join(f"- {t}" for t in draft.commercial_block.solved_tasks)
    adv_md = "\n".join(f"- {a}" for a in draft.commercial_block.key_advantages)

    missing_md = (
        "\n".join(f"- ⚠️ {m}" for m in draft.completeness.missing_fields)
        if draft.completeness.missing_fields
        else "- Все ключевые поля заполнены."
    )
    clarif_md = (
        "\n".join(f"- ❓ {c}" for c in draft.completeness.clarification_questions)
        if draft.completeness.clarification_questions
        else "- Дополнительных уточнений не требуется."
    )

    md = (
        f"# {draft.seo_title}\n\n"
        f"**Длина SEO-заголовка:** {draft.seo_title_length} / 70 симв. | "
        f"**Длина SEO-описания:** {draft.seo_description_length} симв. (норма 800–1200)\n\n"
        f"## SEO-описание\n\n{draft.seo_description}\n\n"
        f"## Технические характеристики\n\n"
        f"| Параметр | Значение |\n|---|---|\n"
        + "\n".join(specs_md_lines)
        + f"\n\n## Рекомендуемые поисковые запросы (SEO)\n\n{queries_md}\n\n"
        f"## Commercial-блок для менеджера\n\n"
        f"### Кому подходит прибор\n{aud_md}\n\n"
        f"### Какие задачи решает\n{tasks_md}\n\n"
        f"### Ключевые аргументы для КП\n{adv_md}\n\n"
        f"## Контроль полноты данных (оценка: {draft.completeness.completeness_score}/100)\n\n"
        f"### Отсутствующие поля\n{missing_md}\n\n"
        f"### Требуют уточнения\n{clarif_md}\n"
    )

    txt = (
        f"SEO-ЗАГОЛОВОК ({draft.seo_title_length} симв.):\n{draft.seo_title}\n\n"
        f"SEO-ОПИСАНИЕ ({draft.seo_description_length} симв.):\n{draft.seo_description}\n\n"
        f"ТЕХНИЧЕСКИЕ ХАРАКТЕРИСТИКИ:\n" + "\n".join(specs_text_lines) + "\n\n"
        f"ПОИСКОВЫЕ ЗАПРОСЫ:\n{queries_txt}\n\n"
        f"COMMERCIAL-БЛОК ДЛЯ МЕНЕДЖЕРА:\n"
        f"Кому подходит:\n" + "\n".join(f"• {a}" for a in draft.commercial_block.target_audience) + "\n"
        f"Какие задачи решает:\n" + "\n".join(f"• {t}" for t in draft.commercial_block.solved_tasks) + "\n"
        f"Аргументы для КП:\n" + "\n".join(f"• {a}" for a in draft.commercial_block.key_advantages) + "\n\n"
        f"КОНТРОЛЬ ПОЛНОТЫ ДАННЫХ ({draft.completeness.completeness_score}/100):\n"
        + (
            "Отсутствующие поля: " + "; ".join(draft.completeness.missing_fields)
            if draft.completeness.missing_fields
            else "Все ключевые поля присутствуют."
        )
    )

    return md, txt
