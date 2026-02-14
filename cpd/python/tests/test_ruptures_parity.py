import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import parity_harness as ph

if importlib.util.find_spec("ruptures") is None:
    pytest.skip("ruptures is required for parity tests", allow_module_level=True)

import cpd as _cpd_import_check  # noqa: F401


MANIFEST_PATH = TESTS_DIR / "fixtures" / "parity" / "corpus_manifest.v1.json"


def _required_category_minimums() -> dict[str, int]:
    return {
        "synthetic_step": 18,
        "synthetic_volatility": 8,
        "heavy_tailed": 8,
        "autocorrelated": 6,
        "trending": 6,
        "missing_gap": 4,
        "real_world_subset": 6,
    }


def test_ruptures_parity_contract() -> None:
    cases = ph.load_manifest(MANIFEST_PATH)
    assert len(cases) >= 56

    counts = ph.category_counts(cases)
    for category, minimum in _required_category_minimums().items():
        assert counts.get(category, 0) >= minimum, f"insufficient cases for {category}"

    profile = os.environ.get("CPD_PARITY_PROFILE", "smoke").strip().lower()
    selected = ph.select_cases(cases, profile=profile)
    assert selected, f"no parity cases selected for profile={profile!r}"

    results = ph.run_parity_suite(selected, manifest_path=MANIFEST_PATH)
    assert results, "parity suite returned no results"

    # Cost parity gate: for exact-change-point matches, relative error must be <= 1e-6.
    for result in results:
        if result.exact_match:
            assert (
                result.cost_rel_error <= 1e-6
            ), f"exact-match cost mismatch case={result.case_id} detector={result.detector} model={result.model}: {result.cost_rel_error}"

    summary = ph.summarize_results(results)
    thresholds = {
        "pelt:l2": 0.95,
        "binseg:l2": 0.90,
        "pelt:normal": 0.90,
        "binseg:normal": 0.90,
    }

    for key, threshold in thresholds.items():
        assert key in summary, f"missing parity summary bucket {key}"
        rate = summary[key]["tolerant_rate"]
        assert rate >= threshold, f"{key} tolerant_rate={rate:.4f} < {threshold:.4f}"

    report_out = os.environ.get("CPD_PARITY_REPORT_OUT")
    if report_out:
        payload = {
            "profile": profile,
            "manifest_path": str(MANIFEST_PATH),
            "summary": summary,
            "results": ph.results_to_jsonable(results),
        }
        out_path = Path(report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
