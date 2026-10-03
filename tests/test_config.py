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
      default: NYTK/sentiment-hts5-xlm-roberta-hungarian
      candidates:
        - NYTK/sentiment-hts5-xlm-roberta-hungarian
        - NYTK/sentiment-ohb3-xlm-roberta-hungarian
        - cardiffnlp/twitter-xlm-roberta-base-sentiment
    scraping: { delay_sec: 0.35, retries: 3 }
    scoring:
      backend: jev
      model: jev-latest
      concurrency: 32
      max_consecutive_failures: 10
      cost_cap_usd: 9.0
      request_delay_sec: 0.2
      retries: 3
      flush_every: 500
    aggregation: { weight_scheme: engagement, z_window: 60, z_min_periods: 30 }
    calibration: { features: [score_wmean], min_warmup: 120 }
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
    assert len(cfg.models.candidates) == 3
    assert cfg.models.candidates[0] == "NYTK/sentiment-hts5-xlm-roberta-hungarian"
    assert cfg.models.default == "NYTK/sentiment-hts5-xlm-roberta-hungarian"
    assert cfg.models.default in cfg.models.candidates
    assert cfg.scraping.delay_sec == 0.35
    assert cfg.scraping.retries == 3
    assert cfg.scoring.backend == "jev"
    assert cfg.scoring.model == "jev-latest"
    assert cfg.scoring.concurrency == 32
    assert cfg.scoring.max_consecutive_failures == 10
    assert cfg.scoring.cost_cap_usd == 9.0
    assert cfg.scoring.request_delay_sec == 0.2
    assert cfg.scoring.retries == 3
    assert cfg.scoring.flush_every == 500
    assert cfg.aggregation.weight_scheme == "engagement"
    assert cfg.aggregation.z_window == 60
    assert cfg.aggregation.z_min_periods == 30
    assert cfg.calibration.features == ("score_wmean",)
    assert cfg.calibration.min_warmup == 120
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


def test_models_candidates_must_be_non_empty_list(tmp_path):
    text = VALID_YAML.replace(
        "  candidates:\n"
        "    - NYTK/sentiment-hts5-xlm-roberta-hungarian\n"
        "    - NYTK/sentiment-ohb3-xlm-roberta-hungarian\n"
        "    - cardiffnlp/twitter-xlm-roberta-base-sentiment\n",
        "  candidates: []\n",
    )
    with pytest.raises(ConfigError, match="models.candidates: expected a non-empty list"):
        load_config(_write(tmp_path, text))


def test_models_default_must_be_a_candidate(tmp_path):
    text = VALID_YAML.replace(
        "  default: NYTK/sentiment-hts5-xlm-roberta-hungarian",
        "  default: some/other-model",
    )
    with pytest.raises(ConfigError, match="models.default .* must be one of"):
        load_config(_write(tmp_path, text))


def test_models_missing_default_raises(tmp_path):
    text = VALID_YAML.replace(
        "  default: NYTK/sentiment-hts5-xlm-roberta-hungarian\n", ""
    )
    with pytest.raises(ConfigError, match="models: missing required key 'default'"):
        load_config(_write(tmp_path, text))


def test_root_not_a_mapping_raises(tmp_path):
    with pytest.raises(ConfigError, match="expected a mapping"):
        load_config(_write(tmp_path, "- just\n- a\n- list\n"))


def test_env_override(tmp_path, monkeypatch):
    path = _write(tmp_path, VALID_YAML)
    monkeypatch.setenv("PRESSFLOW_CONFIG", str(path))
    assert get_config().window_years == 10


# --- scoring section (T3.4) ------------------------------------------------


def test_scoring_missing_section_raises(tmp_path):
    block = (
        "scoring:\n"
        "  backend: jev\n"
        "  model: jev-latest\n"
        "  concurrency: 32\n"
        "  max_consecutive_failures: 10\n"
        "  cost_cap_usd: 9.0\n"
        "  request_delay_sec: 0.2\n"
        "  retries: 3\n"
        "  flush_every: 500\n"
    )
    with pytest.raises(ConfigError, match="missing required key 'scoring'"):
        load_config(_write(tmp_path, VALID_YAML.replace(block, "")))


@pytest.mark.parametrize("bad", [0, -1])
def test_scoring_concurrency_not_positive_raises(tmp_path, bad):
    text = VALID_YAML.replace("concurrency: 32", f"concurrency: {bad}")
    with pytest.raises(ConfigError, match="scoring.concurrency: expected a positive"):
        load_config(_write(tmp_path, text))


@pytest.mark.parametrize("bad", [0, -2])
def test_scoring_max_consecutive_failures_not_positive_raises(tmp_path, bad):
    text = VALID_YAML.replace(
        "max_consecutive_failures: 10", f"max_consecutive_failures: {bad}"
    )
    with pytest.raises(
        ConfigError, match="scoring.max_consecutive_failures: expected a positive"
    ):
        load_config(_write(tmp_path, text))


@pytest.mark.parametrize("bad", [0, -1.0])
def test_scoring_cost_cap_not_positive_raises(tmp_path, bad):
    text = VALID_YAML.replace("cost_cap_usd: 9.0", f"cost_cap_usd: {bad}")
    with pytest.raises(ConfigError, match="scoring.cost_cap_usd: expected a positive"):
        load_config(_write(tmp_path, text))


@pytest.mark.parametrize("bad", [-0.1])
def test_scoring_request_delay_negative_raises(tmp_path, bad):
    text = VALID_YAML.replace("request_delay_sec: 0.2", f"request_delay_sec: {bad}")
    with pytest.raises(
        ConfigError, match="scoring.request_delay_sec: expected a non-negative"
    ):
        load_config(_write(tmp_path, text))


@pytest.mark.parametrize("bad", [0, -1])
def test_scoring_retries_not_positive_raises(tmp_path, bad):
    text = VALID_YAML.replace("\n  retries: 3", f"\n  retries: {bad}")
    with pytest.raises(ConfigError, match="scoring.retries: expected a positive"):
        load_config(_write(tmp_path, text))


def test_scoring_request_delay_allows_zero(tmp_path):
    text = VALID_YAML.replace("request_delay_sec: 0.2", "request_delay_sec: 0.0")
    cfg = load_config(_write(tmp_path, text))
    assert cfg.scoring.request_delay_sec == 0.0


# --- aggregation section (T4.1) --------------------------------------------


AGG_BLOCK = "aggregation: { weight_scheme: engagement, z_window: 60, z_min_periods: 30 }\n"


def test_aggregation_missing_section_raises(tmp_path):
    with pytest.raises(ConfigError, match="missing required key 'aggregation'"):
        load_config(_write(tmp_path, VALID_YAML.replace(AGG_BLOCK, "")))


@pytest.mark.parametrize("bad", [0, -1])
def test_aggregation_z_window_not_positive_raises(tmp_path, bad):
    text = VALID_YAML.replace("z_window: 60", f"z_window: {bad}")
    with pytest.raises(
        ConfigError, match="aggregation.z_window: expected a positive"
    ):
        load_config(_write(tmp_path, text))


@pytest.mark.parametrize("bad", [0, -1, 61])
def test_aggregation_z_min_periods_out_of_range_raises(tmp_path, bad):
    text = VALID_YAML.replace("z_min_periods: 30", f"z_min_periods: {bad}")
    with pytest.raises(ConfigError, match="aggregation.z_min_periods: expected an integer"):
        load_config(_write(tmp_path, text))


def test_aggregation_weight_scheme_must_be_string(tmp_path):
    text = VALID_YAML.replace("weight_scheme: engagement", "weight_scheme: 7")
    with pytest.raises(
        ConfigError, match="aggregation.weight_scheme: expected a non-empty string"
    ):
        load_config(_write(tmp_path, text))


# --- calibration section (T4.4) --------------------------------------------

CAL_BLOCK = "calibration: { features: [score_wmean], min_warmup: 120 }\n"


def test_calibration_missing_section_raises(tmp_path):
    with pytest.raises(ConfigError, match="missing required key 'calibration'"):
        load_config(_write(tmp_path, VALID_YAML.replace(CAL_BLOCK, "")))


def test_calibration_features_empty_raises(tmp_path):
    text = VALID_YAML.replace("features: [score_wmean]", "features: []")
    with pytest.raises(
        ConfigError, match="calibration.features: expected a non-empty list"
    ):
        load_config(_write(tmp_path, text))


def test_calibration_features_not_a_list_raises(tmp_path):
    text = VALID_YAML.replace("features: [score_wmean]", "features: score_wmean")
    with pytest.raises(
        ConfigError, match="calibration.features: expected a non-empty list"
    ):
        load_config(_write(tmp_path, text))


def test_calibration_feature_must_be_string(tmp_path):
    text = VALID_YAML.replace("features: [score_wmean]", "features: [7]")
    with pytest.raises(
        ConfigError, match=r"calibration.features\[0\]: expected a non-empty string"
    ):
        load_config(_write(tmp_path, text))


def test_calibration_multiple_features(tmp_path):
    text = VALID_YAML.replace(
        "features: [score_wmean]", "features: [score_wmean, z_score]"
    )
    cfg = load_config(_write(tmp_path, text))
    assert cfg.calibration.features == ("score_wmean", "z_score")


@pytest.mark.parametrize("bad", [0, -1])
def test_calibration_min_warmup_not_positive_raises(tmp_path, bad):
    text = VALID_YAML.replace("min_warmup: 120", f"min_warmup: {bad}")
    with pytest.raises(
        ConfigError, match="calibration.min_warmup: expected a positive integer"
    ):
        load_config(_write(tmp_path, text))
