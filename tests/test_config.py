"""Tests for the validated config loader (T0.2)."""

from __future__ import annotations

import textwrap

import pytest

from pressflow.config import ConfigError, get_config, load_config

VALID_YAML = textwrap.dedent(
    """
    forum:
      base_url: https://forum.portfolio.hu/topics/otp-reszvenyesek-ide/5224
      thread_id: 5224
    window_years: 10
    costs: { base: 0.003, low: 0.0015, high: 0.006 }
    kelly: { k: 0.25, shrink_gamma: 0.5, cap: 1.0 }
    models:
      - NYTK/sentiment-hts5-xlm-roberta-hungarian
      - NYTK/sentiment-ohb3-xlm-roberta-hungarian
      - cardiffnlp/twitter-xlm-roberta-base-sentiment
    scraping: { delay_sec: 0.35, retries: 3 }
    paths: { data: data, raw: data/raw, processed: data/processed }
    """
)


def _write(tmp_path, text: str):
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_default_config_loads():
    cfg = get_config()
    assert cfg.forum.thread_id == 5224
    assert cfg.forum.base_url.endswith("/5224")
    assert cfg.window_years == 10
    assert cfg.costs.base == 0.003
    assert cfg.costs.low == 0.0015
    assert cfg.costs.high == 0.006
    assert cfg.kelly.k == 0.25
    assert cfg.kelly.shrink_gamma == 0.5
    assert cfg.kelly.cap == 1.0
    assert len(cfg.models) == 3
    assert cfg.models[0] == "NYTK/sentiment-hts5-xlm-roberta-hungarian"
    assert cfg.scraping.delay_sec == 0.35
    assert cfg.scraping.retries == 3
    assert cfg.paths.data == "data"
    assert cfg.paths.raw == "data/raw"
    assert cfg.paths.processed == "data/processed"


def test_valid_config_loads_from_path(tmp_path):
    cfg = load_config(_write(tmp_path, VALID_YAML))
    assert cfg.costs.base == 0.003


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does-not-exist.yaml")


def test_missing_key_raises(tmp_path):
    text = VALID_YAML.replace("window_years: 10\n", "")
    with pytest.raises(ConfigError, match="missing required key 'window_years'"):
        load_config(_write(tmp_path, text))


@pytest.mark.parametrize("bad", [0.0, -0.001, 0.02, 0.05])
def test_cost_out_of_range_raises(tmp_path, bad):
    text = VALID_YAML.replace("base: 0.003", f"base: {bad}")
    with pytest.raises(ConfigError, match="costs.base: expected value in"):
        load_config(_write(tmp_path, text))


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1])
def test_kelly_k_out_of_range_raises(tmp_path, bad):
    text = VALID_YAML.replace("k: 0.25", f"k: {bad}")
    with pytest.raises(ConfigError, match="kelly.k: expected value in"):
        load_config(_write(tmp_path, text))


@pytest.mark.parametrize("bad", [0, -3])
def test_window_years_not_positive_raises(tmp_path, bad):
    text = VALID_YAML.replace("window_years: 10", f"window_years: {bad}")
    with pytest.raises(ConfigError, match="window_years: expected a positive integer"):
        load_config(_write(tmp_path, text))


@pytest.mark.parametrize("bad", [0, -0.5])
def test_delay_sec_not_positive_raises(tmp_path, bad):
    text = VALID_YAML.replace("delay_sec: 0.35", f"delay_sec: {bad}")
    with pytest.raises(ConfigError, match="scraping.delay_sec: expected a positive value"):
        load_config(_write(tmp_path, text))


def test_wrong_type_raises(tmp_path):
    text = VALID_YAML.replace("thread_id: 5224", 'thread_id: "5224"')
    with pytest.raises(ConfigError, match="forum.thread_id: expected an integer"):
        load_config(_write(tmp_path, text))


def test_models_must_be_non_empty_list(tmp_path):
    text = VALID_YAML.replace(
        "models:\n"
        "  - NYTK/sentiment-hts5-xlm-roberta-hungarian\n"
        "  - NYTK/sentiment-ohb3-xlm-roberta-hungarian\n"
        "  - cardiffnlp/twitter-xlm-roberta-base-sentiment\n",
        "models: []\n",
    )
    with pytest.raises(ConfigError, match="models: expected a non-empty list"):
        load_config(_write(tmp_path, text))


def test_root_not_a_mapping_raises(tmp_path):
    with pytest.raises(ConfigError, match="expected a mapping"):
        load_config(_write(tmp_path, "- just\n- a\n- list\n"))


def test_env_override(tmp_path, monkeypatch):
    path = _write(tmp_path, VALID_YAML)
    monkeypatch.setenv("PRESSFLOW_CONFIG", str(path))
    assert get_config().window_years == 10
