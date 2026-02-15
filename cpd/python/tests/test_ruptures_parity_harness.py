import json
from pathlib import Path
import sys

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import parity_harness as ph


MANIFEST_PATH = TESTS_DIR / "fixtures" / "parity" / "corpus_manifest.v1.json"


def test_manifest_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    manifest = {
        "manifest_version": 1,
        "cases": [
            {
                "id": "dup",
                "category": "synthetic_step",
                "source": "synthetic",
                "expected_behavior": "x",
                "signal_spec": {"kind": "piecewise_constant", "segments": [{"length": 5, "mean": 0.0}]},
                "constraints": {"min_segment_len": 1, "jump": 1},
                "stopping": {"n_bkps": 1},
                "enabled_detectors": ["pelt"],
                "enabled_models": ["l2"],
                "tolerance": {"index": 2},
                "profile_tags": ["smoke"],
            },
            {
                "id": "dup",
                "category": "synthetic_step",
                "source": "synthetic",
                "expected_behavior": "x",
                "signal_spec": {"kind": "piecewise_constant", "segments": [{"length": 5, "mean": 0.0}]},
                "constraints": {"min_segment_len": 1, "jump": 1},
                "stopping": {"n_bkps": 1},
                "enabled_detectors": ["pelt"],
                "enabled_models": ["l2"],
                "tolerance": {"index": 2},
                "profile_tags": ["smoke"],
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ph.ManifestValidationError, match="duplicate case id"):
        ph.load_manifest(path)


def test_tolerance_and_jaccard_behavior() -> None:
    tolerant, matches, jaccard = ph.compare_change_points((10, 20), (11, 19), tolerance=2)
    assert tolerant
    assert matches == 2
    assert jaccard == pytest.approx(1.0)

    tolerant2, matches2, jaccard2 = ph.compare_change_points((10, 20), (50,), tolerance=2)
    assert not tolerant2
    assert matches2 == 0
    assert jaccard2 == pytest.approx(0.0)


def test_signal_generation_is_deterministic() -> None:
    cases = ph.load_manifest(MANIFEST_PATH)
    case = next(c for c in cases if c.id == "step_01")

    a = ph.generate_signal(case, manifest_path=MANIFEST_PATH)
    b = ph.generate_signal(case, manifest_path=MANIFEST_PATH)

    np.testing.assert_allclose(a, b)


def test_missing_preprocess_is_deterministic_and_nan_free() -> None:
    values = np.array([np.nan, np.nan, 1.0, np.nan, 3.0, np.nan], dtype=np.float64)

    first = ph.deterministic_missing_preprocess(values)
    second = ph.deterministic_missing_preprocess(values)

    np.testing.assert_allclose(first, second)
    assert not np.isnan(first).any()
    np.testing.assert_allclose(first, np.array([1.0, 1.0, 1.0, 1.0, 3.0, 3.0]))


def test_cost_evaluator_matches_known_segments() -> None:
    x = np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float64)
    breakpoints = [2, 4]

    l2_cost = ph.evaluate_segmentation_cost(x, breakpoints, model="l2")
    normal_cost = ph.evaluate_segmentation_cost(x, breakpoints, model="normal")

    assert l2_cost == pytest.approx(0.0)
    assert np.isfinite(normal_cost)
    assert normal_cost < 0.0


def test_select_cases_supports_smoke_and_full() -> None:
    cases = ph.load_manifest(MANIFEST_PATH)
    smoke = ph.select_cases(cases, profile="smoke")
    full = ph.select_cases(cases, profile="full")

    assert smoke
    assert full
    assert len(full) >= len(smoke)


def test_run_ruptures_case_pelt_known_k_uses_penalty_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakePelt:
        def __init__(self, model: str, min_size: int, jump: int) -> None:
            self.calls: list[float] = []

        def fit(self, values: np.ndarray) -> "_FakePelt":
            return self

        def predict(self, pen: float) -> list[int]:
            self.calls.append(float(pen))
            if pen < 0.5:
                return [2, 4, 6]
            if pen < 2.0:
                return [4, 6]
            return [6]

    class _FakeBinseg:
        def __init__(self, model: str, min_size: int, jump: int) -> None:
            pass

        def fit(self, values: np.ndarray) -> "_FakeBinseg":
            return self

        def predict(self, n_bkps: int | None = None, pen: float | None = None) -> list[int]:
            _ = (n_bkps, pen)
            return [6]

    class _FakeRuptures:
        Pelt = _FakePelt
        Binseg = _FakeBinseg

    monkeypatch.setattr(ph.importlib, "import_module", lambda name: _FakeRuptures if name == "ruptures" else None)

    values = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float64)
    breakpoints, runtime_ms = ph.run_ruptures_case(
        values=values,
        detector="pelt",
        model="l2",
        constraints={"min_segment_len": 2, "jump": 1},
        stopping={"n_bkps": 1},
    )

    assert breakpoints == [4, 6]
    assert runtime_ms >= 0.0


def test_run_cpd_case_pelt_known_k_uses_penalty_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResult:
        def __init__(self, breakpoints: list[int]) -> None:
            self.breakpoints = breakpoints

    class _FakePelt:
        def __init__(self, model: str, min_segment_len: int, jump: int) -> None:
            self.calls: list[float] = []

        def fit(self, values: np.ndarray) -> "_FakePelt":
            return self

        def predict(
            self, n_bkps: int | None = None, pen: float | None = None
        ) -> _FakeResult:
            if n_bkps is not None:
                raise AssertionError("cpd parity path should not call predict(n_bkps=...)")
            assert pen is not None
            self.calls.append(float(pen))
            if pen < 0.5:
                return _FakeResult([2, 4, 6])
            if pen < 2.0:
                return _FakeResult([4, 6])
            return _FakeResult([6])

    class _FakeBinseg:
        def __init__(self, model: str, min_segment_len: int, jump: int) -> None:
            pass

        def fit(self, values: np.ndarray) -> "_FakeBinseg":
            return self

        def predict(
            self, n_bkps: int | None = None, pen: float | None = None
        ) -> _FakeResult:
            _ = (n_bkps, pen)
            return _FakeResult([6])

    class _FakeCpd:
        Pelt = _FakePelt
        Binseg = _FakeBinseg

    monkeypatch.setattr(ph.importlib, "import_module", lambda name: _FakeCpd if name == "cpd" else None)

    values = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float64)
    breakpoints, runtime_ms = ph.run_cpd_case(
        values=values,
        detector="pelt",
        model="l2",
        constraints={"min_segment_len": 2, "jump": 1},
        stopping={"n_bkps": 1},
    )

    assert breakpoints == [4, 6]
    assert runtime_ms >= 0.0


def test_run_cpd_case_pelt_known_k_returns_best_bracket_when_exact_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResult:
        def __init__(self, breakpoints: list[int]) -> None:
            self.breakpoints = breakpoints

    class _FakePelt:
        def __init__(self, model: str, min_segment_len: int, jump: int) -> None:
            pass

        def fit(self, values: np.ndarray) -> "_FakePelt":
            return self

        def predict(
            self, n_bkps: int | None = None, pen: float | None = None
        ) -> _FakeResult:
            if n_bkps is not None:
                raise AssertionError("cpd parity path should not call predict(n_bkps=...)")
            assert pen is not None
            if pen < 0.5:
                return _FakeResult([1, 2, 4, 6])  # 3 changes
            return _FakeResult([6])  # 0 changes

    class _FakeCpd:
        Pelt = _FakePelt
        Binseg = _FakePelt

    monkeypatch.setattr(ph.importlib, "import_module", lambda name: _FakeCpd if name == "cpd" else None)

    values = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float64)
    breakpoints, runtime_ms = ph.run_cpd_case(
        values=values,
        detector="pelt",
        model="l2",
        constraints={"min_segment_len": 2, "jump": 1},
        stopping={"n_bkps": 2},
    )

    assert breakpoints == [1, 2, 4, 6]
    assert runtime_ms >= 0.0
