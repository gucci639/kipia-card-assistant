"""CLI-утилита для генерации карточек товаров КИПиА из JSON-файлов или YML-фида Союз-Прибор."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from app.feed_importer import fetch_offer_from_yml_feed, fetch_product_from_url
from app.llm_service import LLMServiceError, ProductCardGenerator
from app.schemas import ProductInput


async def run_cli(args: argparse.Namespace) -> int:
    generator = ProductCardGenerator()
    products_to_process: list[tuple[str, ProductInput]] = []

    if args.yml_id:
        try:
            if args.yml_id.startswith(("http://", "https://")):
                prod = await fetch_product_from_url(args.yml_id)
            else:
                prod = await fetch_offer_from_yml_feed(args.yml_id, enrich_from_web=not args.no_enrich)
            products_to_process.append((args.yml_id, prod))
        except Exception as exc:
            print(f"[ОШИБКА ИМПОРТА] {exc}", file=sys.stderr)
            return 1
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[ОШИБКА] Входной файл не найден: {input_path}", file=sys.stderr)
            return 1
        try:
            raw_json = json.loads(input_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[ОШИБКА JSON] Файл {input_path} содержит некорректный JSON: {exc}", file=sys.stderr)
            return 1

        items = raw_json if isinstance(raw_json, list) else [raw_json]
        for idx, entry in enumerate(items):
            if isinstance(entry, dict) and "product" in entry and isinstance(entry["product"], dict):
                item_id = str(entry.get("id") or f"item_{idx + 1}")
                payload = entry["product"]
            elif isinstance(entry, dict):
                item_id = f"item_{idx + 1}"
                payload = entry
            else:
                print(f"[ОШИБКА СХЕМЫ] Элемент #{idx + 1} не является JSON-объектом.", file=sys.stderr)
                return 1

            try:
                prod = ProductInput.model_validate(payload)
                products_to_process.append((item_id, prod))
            except ValidationError as exc:
                print(f"[ОШИБКА ВАЛИДАЦИИ] Элемент {item_id}: {exc}", file=sys.stderr)
                return 1

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for item_id, prod in products_to_process:
        print(f"==> Генерация карточки для [{item_id}]: {prod.name or '(без названия)'} ...", file=sys.stderr)
        try:
            draft = await generator.generate_card(prod)
        except LLMServiceError as exc:
            print(f"[ОШИБКА LLM API] ({exc.status_code}): {exc.message}", file=sys.stderr)
            return 1

        if args.format == "markdown":
            rendered = draft.markdown_export
            ext = "md"
        elif args.format == "text":
            rendered = draft.text_export
            ext = "txt"
        else:
            dump = draft.model_dump()
            rendered = json.dumps(dump, ensure_ascii=False, indent=2)
            ext = "json"

        if output_dir:
            out_file = output_dir / f"{item_id}.{ext}"
            out_file.write_text(rendered, encoding="utf-8")
            # Также всегда сохраняем парные .json и .md для удобства проверки
            (output_dir / f"{item_id}.json").write_text(
                json.dumps(draft.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output_dir / f"{item_id}.md").write_text(draft.markdown_export, encoding="utf-8")
            print(
                f"    Готово! Заголовок: {draft.seo_title_length}/70 симв., "
                f"Описание: {draft.seo_description_length} симв., "
                f"Полнота: {draft.completeness.completeness_score}/100 -> {out_file}",
                file=sys.stderr,
            )
        else:
            print(rendered)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI-генератор карточек товаров КИПиА (JSON / Markdown / Text)"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="data/sample_inputs.json",
        help="Путь к входному JSON-файлу с одним или несколькими товарами (по умолчанию data/sample_inputs.json)",
    )
    parser.add_argument(
        "--yml-id",
        default=None,
        help="ID предложения из YML-фида Союз-Прибор (например, 205776 или 54079) или прямой URL товара",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Не дообогащать оффер из YML-фида данными с HTML-страницы товара",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "markdown", "text"],
        default="markdown",
        help="Формат вывода в консоль (json, markdown, text)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Директория для сохранения готовых карточек (.json и .md)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run_cli(args)))


if __name__ == "__main__":
    main()
