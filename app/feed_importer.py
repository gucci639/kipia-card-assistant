"""Импорт данных о приборах из YML-фида (https://www.souz-pribor.ru/yml/any/feed.xml) и HTML-страниц товаров."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from app.schemas import ProductInput

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def parse_souz_pribor_html(html: str, fallback_url: str | None = None) -> ProductInput:
    """Извлекает название, производителя, таблицу характеристик и описание со страницы товара Союз-Прибор."""
    soup = BeautifulSoup(html, "lxml")

    # 1. Название из H1
    h1 = soup.find("h1")
    name = re.sub(r"\s+", " ", h1.get_text(" ", strip=True)) if h1 else ""

    # 2. Производитель
    manufacturer = ""
    # Ищем ссылку/блок производителя рядом с карточкой или в хлебных крошках
    for text_node in soup.find_all(string=re.compile(r"\(Россия\)|\(Германия\)|\(Китай\)|\(Беларусь\)")):
        candidate = re.sub(r"\s*\([^)]+\)\s*", "", text_node.strip()).strip()
        if 2 <= len(candidate) <= 40:
            manufacturer = candidate
            break

    if not manufacturer and name:
        for known in ("ОВЕН", "Манотомь", "Метран", "Росма", "Физтех", "Теплоконтроль", "Элемер", "WIKA", "Testo"):
            if known.lower() in name.lower():
                manufacturer = known
                break

    specs: dict[str, str] = {}

    # 3. Верхний блок кратких характеристик (characteristics__item)
    for item in soup.select(".characteristics__item"):
        divs = item.find_all("div", recursive=False)
        if len(divs) >= 2:
            k = re.sub(r"[:\s]+$", "", divs[0].get_text(" ", strip=True))
            v = re.sub(r"\s+", " ", divs[1].get_text(" ", strip=True))
            if k and v:
                specs[k] = v

    # 4. Полная вкладка характеристик (#specifications)
    spec_block = soup.find(id="specifications")
    if spec_block:
        for tr in spec_block.find_all("tr"):
            cells = [re.sub(r"\s+", " ", td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
            if len(cells) == 2 and cells[0] and cells[1]:
                if cells[0].lower() not in ("наименование", "параметр"):
                    specs[cells[0].rstrip(":")] = cells[1]

    # 5. Текстовое описание (первый блок description-specifications__content-inner без id="specifications")
    description_parts: list[str] = []
    for block in soup.select(".description-specifications__content-inner"):
        if block.get("id") == "specifications":
            continue
        # Удаляем таблицы модификаций/сравнений внутри описания, чтобы оставить чистый текст паспорта
        for tbl in block.find_all("table"):
            tbl.decompose()
        raw_text = block.get_text("\n", strip=True)
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw_text.splitlines() if ln.strip()]
        # Фильтруем технические подписи кнопок ("показать все", "скрыть")
        lines = [ln for ln in lines if ln.lower() not in ("показать все", "скрыть")]
        if lines:
            description_parts.append(" ".join(lines))

    description = "\n\n".join(description_parts).strip()
    if len(description) > 2500:
        description = description[:2500].rsplit(" ", 1)[0] + "..."

    return ProductInput(
        name=name,
        manufacturer=manufacturer,
        specs=specs,
        description=description,
        source_url=fallback_url,
    )


async def fetch_product_from_url(url: str, timeout: float = 20.0) -> ProductInput:
    """Скачивает карточку товара по URL и парсит её в структуру ProductInput."""
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; KipiaCardAssistant/1.0)"},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return parse_souz_pribor_html(resp.text, fallback_url=url)


async def fetch_offer_from_yml_feed(
    offer_id_or_query: str,
    feed_url: str = "https://www.souz-pribor.ru/yml/any/feed.xml",
    enrich_from_web: bool = True,
) -> ProductInput:
    """Находит товар в YML-фиде по ID, артикулу или подстроке названия и при необходимости
    дообогащает характеристиками и описанием с URL страницы товара."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_xml = CACHE_DIR / "souz_pribor_feed.xml"

    if not cached_xml.exists():
        logger.info("Скачивание YML-фида %s ...", feed_url)
        async with httpx.AsyncClient(
            timeout=45.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; KipiaCardAssistant/1.0)"},
        ) as client:
            resp = await client.get(feed_url)
            resp.raise_for_status()
            cached_xml.write_bytes(resp.content)

    query_clean = offer_id_or_query.strip().lower()
    matched_elem: ET.Element | None = None

    # Потоковый парсинг iterparse для экономии памяти на 37 МБ XML
    for _event, elem in ET.iterparse(str(cached_xml), events=("end",)):
        if elem.tag == "offer":
            oid = (elem.attrib.get("id") or "").strip().lower()
            vcode = (elem.findtext("vendorCode") or "").strip().lower()
            name_txt = (elem.findtext("name") or "").strip().lower()
            if query_clean in (oid, vcode) or (len(query_clean) >= 3 and query_clean in name_txt):
                matched_elem = elem
                break
            elem.clear()

    if matched_elem is None:
        raise ValueError(f"Товар по запросу «{offer_id_or_query}» не найден в YML-фиде.")

    offer_id = matched_elem.attrib.get("id", "")
    raw_name = (matched_elem.findtext("name") or "").strip()
    type_prefix = (matched_elem.findtext("typePrefix") or "").strip()
    full_name = f"{type_prefix.capitalize()} {raw_name}".strip() if type_prefix else raw_name
    vendor = (matched_elem.findtext("vendor") or "").strip()
    page_url = (matched_elem.findtext("url") or "").strip()
    price = (matched_elem.findtext("price") or "").strip()
    currency = (matched_elem.findtext("currencyId") or "RUR").strip()

    yml_specs: dict[str, str] = {}
    if offer_id:
        yml_specs["ID предложения (YML)"] = offer_id
    for param in matched_elem.findall("param"):
        p_name = (param.attrib.get("name") or "").strip()
        p_val = (param.text or "").strip()
        if p_name and p_val:
            yml_specs[p_name] = p_val
    if price:
        yml_specs["Цена в фиде"] = f"{price} {currency}"

    yml_desc = (matched_elem.findtext("description") or "").strip()

    if enrich_from_web and page_url:
        try:
            web_product = await fetch_product_from_url(page_url)
            merged_specs = {**yml_specs, **web_product.specs}
            return ProductInput(
                name=web_product.name or full_name,
                manufacturer=vendor or web_product.manufacturer,
                specs=merged_specs,
                description=web_product.description or yml_desc,
                source_url=page_url,
            )
        except Exception as exc:
            logger.warning("Не удалось обогатить карточку по URL %s: %s", page_url, exc)

    return ProductInput(
        name=full_name,
        manufacturer=vendor,
        specs=yml_specs,
        description=yml_desc,
        source_url=page_url or feed_url,
    )
