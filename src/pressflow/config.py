"""Validated, config-driven parameters for the whole pipeline (ADR-0015).

Every downstream module reads its parameters from here instead of hardcoding
them, so the values in ``config/default.yaml`` are the single source of truth.

Usage::

    from pressflow.config import get_config

    cfg = get_config()
    cfg.costs.base        # 0.003
    cfg.kelly.k           # 0.25
    cfg.window_years      # 10

Invalid or missing values raise :class:`ConfigError` with a clear message.
The loader depends only on ``pyyaml`` and the standard library.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: Repository root, resolved from this file (``<root>/src/pressflow/config.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default config location.
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"

#: Environment variable that overrides the config path.
ENV_VAR = "PRESSFLOW_CONFIG"

#: Allowed inclusive bounds for round-trip costs (ADR-0012).
COST_MIN = 0.0  # exclusive
COST_MAX = 0.02  # exclusive


class ConfigError(ValueError):
    """Raised when the configuration is missing, malformed or out of range."""


# --- typed sections --------------------------------------------------------


@dataclass(frozen=True)
class Forum:
    base_url: str
    thread_id: int


@dataclass(frozen=True)
class Costs:
    base: float
    low: float
    high: float


@dataclass(frozen=True)
class Kelly:
    k: float
    shrink_gamma: float
    cap: float


@dataclass(frozen=True)
class Scraping:
    delay_sec: float
    retries: int


@dataclass(frozen=True)
class Paths:
    data: str
    raw: str
    processed: str


@dataclass(frozen=True)
class Config:
    forum: Forum
    window_years: int
    costs: Costs
    kelly: Kelly
    models: tuple[str, ...]
    scraping: Scraping
    paths: Paths


# --- validation helpers ----------------------------------------------------


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: expected a number, got {type(value).__name__}")
    return float(value)


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: expected an integer, got {type(value).__name__}")
    return value


def _as_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where}: expected a non-empty string, got {type(value).__name__}")
    return value


def _in_open_range(value: float, low: float, high: float, where: str) -> float:
    if not (low < value < high):
        raise ConfigError(f"{where}: expected value in ({low}, {high}), got {value}")
    return value


def _positive(value: float, where: str) -> float:
    if value <= 0:
        raise ConfigError(f"{where}: expected a positive value, got {value}")
    return value


# --- section parsers -------------------------------------------------------


def _parse_forum(raw: Any) -> Forum:
    m = _mapping(raw, "forum")
    return Forum(
        base_url=_as_str(_require(m, "base_url", "forum"), "forum.base_url"),
        thread_id=_as_int(_require(m, "thread_id", "forum"), "forum.thread_id"),
    )


def _parse_costs(raw: Any) -> Costs:
    m = _mapping(raw, "costs")
    return Costs(
        base=_in_open_range(
            _as_float(_require(m, "base", "costs"), "costs.base"),
            COST_MIN,
            COST_MAX,
            "costs.base",
        ),
        low=_in_open_range(
            _as_float(_require(m, "low", "costs"), "costs.low"),
            COST_MIN,
            COST_MAX,
            "costs.low",
        ),
        high=_in_open_range(
            _as_float(_require(m, "high", "costs"), "costs.high"),
            COST_MIN,
            COST_MAX,
            "costs.high",
        ),
    )


def _parse_kelly(raw: Any) -> Kelly:
    m = _mapping(raw, "kelly")
    return Kelly(
        k=_in_open_range(
            _as_float(_require(m, "k", "kelly"), "kelly.k"), 0.0, 1.0, "kelly.k"
        ),
        shrink_gamma=_in_open_range(
            _as_float(_require(m, "shrink_gamma", "kelly"), "kelly.shrink_gamma"),
            0.0,
            1.0,
            "kelly.shrink_gamma",
        ),
        cap=_positive(
            _as_float(_require(m, "cap", "kelly"), "kelly.cap"), "kelly.cap"
        ),
    )


def _parse_models(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"models: expected a non-empty list, got {type(raw).__name__}")
    return tuple(_as_str(item, f"models[{i}]") for i, item in enumerate(raw))


def _parse_scraping(raw: Any) -> Scraping:
    m = _mapping(raw, "scraping")
    delay = _positive(
        _as_float(_require(m, "delay_sec", "scraping"), "scraping.delay_sec"),
        "scraping.delay_sec",
    )
    retries = _as_int(_require(m, "retries", "scraping"), "scraping.retries")
    if retries < 0:
        raise ConfigError(f"scraping.retries: expected a non-negative integer, got {retries}")
    return Scraping(delay_sec=delay, retries=retries)


def _parse_paths(raw: Any) -> Paths:
    m = _mapping(raw, "paths")
    return Paths(
        data=_as_str(_require(m, "data", "paths"), "paths.data"),
        raw=_as_str(_require(m, "raw", "paths"), "paths.raw"),
        processed=_as_str(_require(m, "processed", "paths"), "paths.processed"),
    )


# --- public API ------------------------------------------------------------


def resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the config path: explicit arg > ``PRESSFLOW_CONFIG`` > default."""
    if path is not None:
        return Path(path)
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env)
    return DEFAULT_CONFIG_PATH


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate a config file.

    Raises :class:`ConfigError` for a missing file, malformed YAML, missing keys
    or out-of-range values.
    """
    resolved = resolve_config_path(path)
    if not resolved.is_file():
        raise ConfigError(f"config file not found: {resolved}")

    try:
        loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - message passthrough
        raise ConfigError(f"invalid YAML in {resolved}: {exc}") from exc

    root = _mapping(loaded, str(resolved))

    window_years = _as_int(_require(root, "window_years", str(resolved)), "window_years")
    if window_years <= 0:
        raise ConfigError(f"window_years: expected a positive integer, got {window_years}")

    return Config(
        forum=_parse_forum(_require(root, "forum", str(resolved))),
        window_years=window_years,
        costs=_parse_costs(_require(root, "costs", str(resolved))),
        kelly=_parse_kelly(_require(root, "kelly", str(resolved))),
        models=_parse_models(_require(root, "models", str(resolved))),
        scraping=_parse_scraping(_require(root, "scraping", str(resolved))),
        paths=_parse_paths(_require(root, "paths", str(resolved))),
    )


def get_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Return the validated configuration (default file unless overridden)."""
    return load_config(path)
