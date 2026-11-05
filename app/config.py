"""Конфигурация приложения через переменные окружения (.env)."""

from __future__ import annotations

import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки подключения к LLM API и внешним источникам данных."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_key: str = Field(
        default="",
        description="API-ключ LLM (LLM_API_KEY или OPENAI_API_KEY)",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Базовый URL OpenAI-совместимого API",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Идентификатор модели LLM",
    )
    llm_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Температура генерации (низкая для исключения галлюцинаций)",
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        gt=1.0,
        description="Таймаут запроса к LLM API в секундах",
    )
    yml_feed_url: str = Field(
        default="https://www.souz-pribor.ru/yml/any/feed.xml",
        description="URL YML-фида каталога приборов",
    )

    def resolved_api_key(self) -> str:
        """Возвращает API-ключ из LLM_API_KEY или стандартных переменных окружения."""
        if self.llm_api_key and self.llm_api_key != "your_api_key_here":
            return self.llm_api_key.strip()
        for env_name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "XKIRO_API_KEY"):
            val = os.getenv(env_name, "").strip()
            if val:
                return val
        return ""


@lru_cache
def get_settings() -> Settings:
    """Возвращает кэшированный объект настроек."""
    return Settings()
