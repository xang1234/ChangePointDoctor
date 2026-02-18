import json
from pathlib import Path

import cpd
import numpy as np
from jsonschema import Draft202012Validator


CPD_ROOT = Path(__file__).resolve().parents[2]
RESULT_SCHEMA_V1_PATH = (
    CPD_ROOT / "schemas" / "result" / "offline_change_point_result.v1.schema.json"
)
RESULT_V1_FIXTURE_PATH = (
    CPD_ROOT / "tests" / "fixtures" / "migrations" / "result" / "offline_result.v1.json"
)
RESULT_V2_FIXTURE_PATH = (
    CPD_ROOT
    / "tests"
    / "fixtures"
    / "migrations"
    / "result"
    / "offline_result.v2.additive.json"
)
PYTHON_V1_FIXTURE_PATH = (
    CPD_ROOT
    / "crates"
    / "cpd-python"
    / "tests"
    / "fixtures"
    / "offline_result_v1.json"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _schema_for_payload(payload: dict) -> dict:
    schema_version = payload.get("diagnostics", {}).get("schema_version")
    if schema_version in (1, 2):
        # v2 is currently additive-policy validated against canonical v1 schema.
        return _load_json(RESULT_SCHEMA_V1_PATH)
    raise AssertionError(f"unsupported diagnostics.schema_version for contract test: {schema_version!r}")


def _validate_payload(payload: dict, *, context: str) -> None:
    validator = Draft202012Validator(_schema_for_payload(payload))
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.absolute_path))
    assert not errors, "\n".join(
        f"{context}: {'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in errors
    )


def _three_regime_signal() -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(40, dtype=np.float64),
            np.full(40, 8.0, dtype=np.float64),
            np.full(40, -4.0, dtype=np.float64),
        ]
    )


def test_result_schema_declares_required_contract_fields() -> None:
    schema = _load_json(RESULT_SCHEMA_V1_PATH)

    assert set(schema["required"]) == {"breakpoints", "change_points", "diagnostics"}
    diagnostics = schema["$defs"]["diagnostics"]
    assert set(diagnostics["required"]) == {
        "n",
        "d",
        "schema_version",
        "algorithm",
        "cost_model",
        "repro_mode",
    }
    assert "build" in diagnostics["properties"]


def test_migration_fixtures_validate_against_version_selected_schema() -> None:
    v1 = _load_json(RESULT_V1_FIXTURE_PATH)
    v2 = _load_json(RESULT_V2_FIXTURE_PATH)

    assert v1["diagnostics"]["schema_version"] == 1
    assert v2["diagnostics"]["schema_version"] == 2

    _validate_payload(v1, context="offline_result.v1.json")
    _validate_payload(v2, context="offline_result.v2.additive.json")


def test_python_result_fixture_matches_canonical_v1_fixture() -> None:
    python_v1 = _load_json(PYTHON_V1_FIXTURE_PATH)
    migration_v1 = _load_json(RESULT_V1_FIXTURE_PATH)

    assert python_v1 == migration_v1
    _validate_payload(python_v1, context="cpd-python offline_result_v1.json")


def test_live_runtime_output_validates_and_exposes_build_context() -> None:
    result = cpd.detect_offline(
        _three_regime_signal(),
        detector="pelt",
        cost="l2",
        constraints={"min_segment_len": 2},
        stopping={"n_bkps": 2},
        repro_mode="balanced",
    )
    payload = json.loads(result.to_json())
    _validate_payload(payload, context="live detect_offline payload")

    build = payload["diagnostics"].get("build")
    assert isinstance(build, dict)
    assert build.get("abi") == "pyo3-abi3-py39"
    assert isinstance(build.get("features"), list)
    assert "serde" in build["features"]


def test_backcompat_fixture_roundtrip_keeps_build_field_omitted_when_absent() -> None:
    fixture_payload = RESULT_V1_FIXTURE_PATH.read_text(encoding="utf-8")
    parsed = cpd.OfflineChangePointResult.from_json(fixture_payload)
    roundtrip = json.loads(parsed.to_json())

    assert "build" not in roundtrip["diagnostics"]
    _validate_payload(roundtrip, context="v1 fixture roundtrip payload")
