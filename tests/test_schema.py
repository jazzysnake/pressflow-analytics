"""Tests that the code schema constants match docs/schema.md and are well typed."""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pytest

from pressflow import schema as s

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DOC = REPO_ROOT / "docs" / "schema.md"

ARTIFACTS = ["posts", "prices", "scores", "signals", "backtest"]

# Expected Arrow type token per column, keyed by artifact then column name.
# "timestamp" is expanded/checked separately so tz + unit are asserted.
EXPECTED_TYPES: dict[str, dict[str, str]] = {
    "posts": {
        "post_id": "int64",
        "hsz_id": "int64",
        "author": "string",
        "user_id": "int64",
        "ts": "timestamp",
        "reply_to_post_id": "int64",
        "reply_to_hsz_id": "int64",
        "upvotes": "int64",
        "downvotes": "int64",
        "text": "string",
        "page": "int64",
    },
    "prices": {
        "date": "timestamp",
        "otp_close_huf": "float64",
        "otp_adj_close_huf": "float64",
        "usdhuf": "float64",
        "spy_adj_close_usd": "float64",
        "spx_close": "float64",
        "otp_usd": "float64",
        "spy_usd": "float64",
    },
    "scores": {
        "hsz_id": "int64",
        "model": "string",
        "p_bull": "float64",
        "p_bear": "float64",
        "p_neutral": "float64",
        "net": "float64",
    },
    "signals": {
        "date": "timestamp",
        "score_wmean": "float64",
        "bear_wmean": "float64",
        "count": "int64",
        "disagreement_std": "float64",
        "entropy": "float64",
        "z_score": "float64",
    },
    "backtest": {
        "date": "timestamp",
        "position": "float64",
        "ret": "float64",
        "cost": "float64",
        "equity": "float64",
    },
}

NULLABLE_COLUMNS = {("posts", "reply_to_post_id"), ("posts", "reply_to_hsz_id")}


def _type_token(arrow_type: pa.DataType) -> str:
    if pa.types.is_timestamp(arrow_type):
        return "timestamp"
    if pa.types.is_integer(arrow_type):
        return "int64"
    if pa.types.is_floating(arrow_type):
        return "float64"
    if pa.types.is_string(arrow_type):
        return "string"
    raise AssertionError(f"unexpected Arrow type: {arrow_type}")


def _documented_columns(artifact: str) -> list[str]:
    """Parse the markdown table under the ``<artifact>`` heading in docs/schema.md."""
    current: str | None = None
    names: list[str] = []
    for line in SCHEMA_DOC.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^#{2,6}\s+(\S+)", line)
        if heading:
            current = heading.group(1)
            continue
        if current == artifact:
            row = re.match(r"^\|\s*`([A-Za-z0-9_]+)`\s*\|", line)
            if row:
                names.append(row.group(1))
    return names


@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_schema_column_set_matches_documented_list(artifact: str) -> None:
    assert list(s.SCHEMAS[artifact].names) == _documented_columns(artifact)


@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_column_constants_match_schema(artifact: str) -> None:
    assert s.COLUMN_NAMES[artifact] == tuple(s.SCHEMAS[artifact].names)


@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_column_types(artifact: str) -> None:
    schema = s.SCHEMAS[artifact]
    for column, token in EXPECTED_TYPES[artifact].items():
        assert _type_token(schema.field(column).type) == token, column


@pytest.mark.parametrize("artifact", ["posts", "prices", "signals", "backtest"])
def test_posts_ts_and_date_are_tz_aware_budapest(artifact: str) -> None:
    """Every temporal column is a Budapest-tz, microsecond timestamp."""
    schema = s.SCHEMAS[artifact]
    temporal = [f.name for f in schema if pa.types.is_timestamp(f.type)]
    assert temporal, f"{artifact} has no temporal column"
    for name in temporal:
        field_type = schema.field(name).type
        assert str(field_type.tz) == s.BUDAPEST_TZ, name
        assert field_type.unit == s.TS_UNIT, name


def test_posts_ts_is_budapest_timestamp() -> None:
    ts_type = s.POSTS_SCHEMA.field("ts").type
    assert pa.types.is_timestamp(ts_type)
    assert str(ts_type.tz) == "Europe/Budapest"
    assert ts_type.unit == "us"


@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_nullable_flags(artifact: str) -> None:
    schema = s.SCHEMAS[artifact]
    for field in schema:
        expected = (artifact, field.name) in NULLABLE_COLUMNS
        assert field.nullable is expected, f"{artifact}.{field.name}"


@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_natural_key_exists_and_is_not_nullable(artifact: str) -> None:
    keys = s.NATURAL_KEYS[artifact]
    assert keys, f"{artifact} has no natural key"
    schema = s.SCHEMAS[artifact]
    for key in keys:
        assert key in schema.names
        assert schema.field(key).nullable is False
    # posts and scores dedupe on hsz_id (T0.3 DoD).
    if artifact in {"posts", "scores"}:
        assert "hsz_id" in keys


@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_schema_is_instantiable(artifact: str) -> None:
    table = s.SCHEMAS[artifact].empty_table()
    assert table.num_rows == 0
    assert table.column_names == list(s.SCHEMAS[artifact].names)
