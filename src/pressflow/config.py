"""Validated, config-driven parameters for the whole pipeline (ADR-0015).

Every downstream module reads its parameters from here instead of hardcoding
them, so the values in ``config/default.yaml`` are the single source of truth.

Usage::

    from pressflow.config import get_config

    cfg = get_config()
    cfg.costs.base        # 0.003
    cfg.kelly.k           # 0.25
    cfg.scoring.backend   # "jev"
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
class Scoring:
    """Batch scoring parameters (T3.4 — concurrent, resumable Jev runs)."""

    backend: str
    model: str
    concurrency: int
    max_consecutive_failures: int
    cost_cap_usd: float
    request_delay_sec: float
    retries: int
    flush_every: int


@dataclass(frozen=True)
class Aggregation:
    """Daily signal aggregation parameters (T4.1 — engagement-weighted)."""

    weight_scheme: str
    z_window: int
    z_min_periods: int


@dataclass(frozen=True)
class Calibration:
    """Walk-forward calibration parameters (T4.4, ADR-0003)."""

    features: tuple[str, ...]
    min_warmup: int


@dataclass(frozen=True)
class Paths:
    data: str
    raw: str
    processed: str


@dataclass(frozen=True)
class Models:
    default: str
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    forum: Forum
    window_years: int
    costs: Costs
    kelly: Kelly
    models: Models
    scraping: Scraping
    scoring: Scoring
    aggregation: Aggregation
    calibration: Calibration
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


def _parse_candidates(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(
            f"models.candidates: expected a non-empty list, got {type(raw).__name__}"
        )
    return tuple(
        _as_str(item, f"models.candidates[{i}]") for i, item in enumerate(raw)
    )


def _parse_models(raw: Any) -> Models:
    m = _mapping(raw, "models")
    candidates = _parse_candidates(_require(m, "candidates", "models"))
    default = _as_str(_require(m, "default", "models"), "models.default")
    if default not in candidates:
        raise ConfigError(
            f"models.default {default!r} must be one of models.candidates "
            f"{list(candidates)}"
        )
    return Models(default=default, candidates=candidates)


def _parse_aggregation(raw: Any) -> Aggregation:
    m = _mapping(raw, "aggregation")
    weight_scheme = _as_str(
        _require(m, "weight_scheme", "aggregation"), "aggregation.weight_scheme"
    )
    z_window = _as_int(_require(m, "z_window", "aggregation"), "aggregation.z_window")
    if z_window < 1:
        raise ConfigError(
            f"aggregation.z_window: expected a positive integer, got {z_window}"
        )
    z_min_periods = _as_int(
        _require(m, "z_min_periods", "aggregation"), "aggregation.z_min_periods"
    )
    if z_min_periods < 1 or z_min_periods > z_window:
        raise ConfigError(
            "aggregation.z_min_periods: expected an integer in "
            f"[1, {z_window}], got {z_min_periods}"
        )
    return Aggregation(
        weight_scheme=weight_scheme,
        z_window=z_window,
        z_min_periods=z_min_periods,
    )


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


def _parse_scoring(raw: Any) -> Scoring:
    m = _mapping(raw, "scoring")
    concurrency = _as_int(
        _require(m, "concurrency", "scoring"), "scoring.concurrency"
    )
    if concurrency < 1:
        raise ConfigError(
            f"scoring.concurrency: expected a positive integer, got {concurrency}"
        )
    max_consecutive = _as_int(
        _require(m, "max_consecutive_failures", "scoring"),
        "scoring.max_consecutive_failures",
    )
    if max_consecutive < 1:
        raise ConfigError(
            "scoring.max_consecutive_failures: expected a positive integer, "
            f"got {max_consecutive}"
        )
    cost_cap = _positive(
        _as_float(
            _require(m, "cost_cap_usd", "scoring"), "scoring.cost_cap_usd"
        ),
        "scoring.cost_cap_usd",
    )
    request_delay = _as_float(
        _require(m, "request_delay_sec", "scoring"), "scoring.request_delay_sec"
    )
    if request_delay < 0:
        raise ConfigError(
            "scoring.request_delay_sec: expected a non-negative value, "
            f"got {request_delay}"
        )
    retries = _as_int(_require(m, "retries", "scoring"), "scoring.retries")
    if retries < 1:
        raise ConfigError(
            f"scoring.retries: expected a positive integer, got {retries}"
        )
    flush_every = _as_int(
        _require(m, "flush_every", "scoring"), "scoring.flush_every"
    )
    if flush_every < 1:
        raise ConfigError(
            f"scoring.flush_every: expected a positive integer, got {flush_every}"
        )
    return Scoring(
        backend=_as_str(_require(m, "backend", "scoring"), "scoring.backend"),
        model=_as_str(_require(m, "model", "scoring"), "scoring.model"),
        concurrency=concurrency,
        max_consecutive_failures=max_consecutive,
        cost_cap_usd=cost_cap,
        request_delay_sec=request_delay,
        retries=retries,
        flush_every=flush_every,
    )


def _parse_calibration(raw: Any) -> Calibration:
    m = _mapping(raw, "calibration")
    features = _require(m, "features", "calibration")
    if not isinstance(features, list) or not features:
        raise ConfigError(
            "calibration.features: expected a non-empty list, "
            f"got {type(features).__name__}"
        )
    parsed = tuple(
        _as_str(item, f"calibration.features[{i}]")
        for i, item in enumerate(features)
    )
    min_warmup = _as_int(
        _require(m, "min_warmup", "calibration"), "calibration.min_warmup"
    )
    if min_warmup < 1:
        raise ConfigError(
            f"calibration.min_warmup: expected a positive integer, got {min_warmup}"
        )
    return Calibration(features=parsed, min_warmup=min_warmup)


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
        scoring=_parse_scoring(_require(root, "scoring", str(resolved))),
        aggregation=_parse_aggregation(_require(root, "aggregation", str(resolved))),
        calibration=_parse_calibration(_require(root, "calibration", str(resolved))),
        paths=_parse_paths(_require(root, "paths", str(resolved))),
    )


def get_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Return the validated configuration (default file unless overridden)."""
    return load_config(path)
