"""Runtime settings loaded from config/config.yaml and environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@dataclass(frozen=True)
class SearchSettings:
    max_results: int = 3
    max_content_chars: int = 900


@dataclass(frozen=True)
class CacheSettings:
    enabled: bool = True
    ttl_seconds: int = 900
    max_entries: int = 256


@dataclass(frozen=True)
class RetrievalSettings:
    max_sources: int = 8


@dataclass(frozen=True)
class AppSettings:
    model_name: str = "openai/gpt-oss-120b"
    search: SearchSettings = field(default_factory=SearchSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)


def _load_yaml() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}

    text = CONFIG_PATH.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if not raw_line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            parsed[current_section] = {}
            continue

        if ":" not in line:
            continue

        key, raw_value = line.split(":", 1)
        value = _coerce_value(raw_value.strip())

        if raw_line.startswith("  ") and current_section:
            parsed[current_section][key.strip()] = value
        else:
            parsed[key.strip()] = value

    return parsed


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.isdigit():
        return int(value)
    return value.strip('"').strip("'")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no"}


def load_settings() -> AppSettings:
    raw = _load_yaml()
    search = raw.get("search", {}) if isinstance(raw.get("search", {}), dict) else {}
    cache = raw.get("cache", {}) if isinstance(raw.get("cache", {}), dict) else {}
    retrieval = (
        raw.get("retrieval", {}) if isinstance(raw.get("retrieval", {}), dict) else {}
    )

    return AppSettings(
        model_name=os.environ.get(
            "ARS_MODEL",
            str(raw.get("model_name", AppSettings.model_name)),
        ),
        search=SearchSettings(
            max_results=int(
                os.environ.get(
                    "ARS_SEARCH_MAX_RESULTS",
                    search.get("max_results", 3),
                )
            ),
            max_content_chars=int(
                os.environ.get(
                    "ARS_SEARCH_MAX_CONTENT_CHARS",
                    search.get("max_content_chars", 900),
                )
            ),
        ),
        cache=CacheSettings(
            enabled=_env_bool(
                "ARS_SEARCH_CACHE_ENABLED",
                bool(cache.get("enabled", True)),
            ),
            ttl_seconds=int(
                os.environ.get(
                    "ARS_SEARCH_CACHE_TTL_SECONDS",
                    cache.get("ttl_seconds", 900),
                )
            ),
            max_entries=int(
                os.environ.get(
                    "ARS_SEARCH_CACHE_MAX_ENTRIES",
                    cache.get("max_entries", 256),
                )
            ),
        ),
        retrieval=RetrievalSettings(
            max_sources=int(
                os.environ.get(
                    "ARS_RETRIEVAL_MAX_SOURCES",
                    retrieval.get("max_sources", 8),
                )
            ),
        ),
    )


SETTINGS = load_settings()
