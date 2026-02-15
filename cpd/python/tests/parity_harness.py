from __future__ import annotations

import importlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

_ALLOWED_DETECTORS = {"pelt", "binseg"}
_ALLOWED_MODELS = {"l2", "normal"}
_ALLOWED_PROFILES = {"smoke", "full"}
_VAR_FLOOR = np.finfo(np.float64).eps * 1e6


@dataclass(frozen=True)
class ParityCase:
    id: str
    category: str
    source: str
    expected_behavior: str
    signal_spec: dict[str, Any]
    constraints: dict[str, Any]
    stopping: dict[str, Any]
    enabled_detectors: tuple[str, ...]
    enabled_models: tuple[str, ...]
    tolerance_index: int
    profile_tags: tuple[str, ...]


@dataclass(frozen=True)
class ParityResult:
    case_id: str
    category: str
    detector: str
    model: str
    tolerance_index: int
    cpd_breakpoints: tuple[int, ...]
    ruptures_breakpoints: tuple[int, ...]
    cpd_change_points: tuple[int, ...]
    ruptures_change_points: tuple[int, ...]
    exact_match: bool
    tolerant_match: bool
    tolerant_matches: int
    tolerant_jaccard: float
    cpd_cost: float
    ruptures_cost: float
    cost_rel_error: float
    cpd_runtime_ms: float
    ruptures_runtime_ms: float


class ManifestValidationError(ValueError):
    pass


def load_manifest(path: Path) -> list[ParityCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ManifestValidationError("manifest root must be an object")

    if payload.get("manifest_version") != 1:
        raise ManifestValidationError(
            f"manifest_version must be 1, got {payload.get('manifest_version')!r}"
        )

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ManifestValidationError("manifest.cases must be a non-empty list")

    out: list[ParityCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ManifestValidationError(f"case at index {index} must be an object")

        case = _parse_case(raw_case, index=index)
        if case.id in seen_ids:
            raise ManifestValidationError(f"duplicate case id {case.id!r}")
        seen_ids.add(case.id)
        out.append(case)

    return out


def _parse_case(raw_case: dict[str, Any], index: int) -> ParityCase:
    required = {
        "id",
        "category",
        "source",
        "expected_behavior",
        "signal_spec",
        "constraints",
        "stopping",
        "enabled_detectors",
        "enabled_models",
        "tolerance",
        "profile_tags",
    }
    missing = sorted(required - set(raw_case))
    if missing:
        raise ManifestValidationError(
            f"case index {index} missing required fields: {', '.join(missing)}"
        )

    case_id = raw_case["id"]
    if not isinstance(case_id, str) or not case_id:
        raise ManifestValidationError(f"case index {index} has invalid id {case_id!r}")

    expected_behavior = raw_case["expected_behavior"]
    if not isinstance(expected_behavior, str) or not expected_behavior.strip():
        raise ManifestValidationError(f"case {case_id!r} must define expected_behavior")

    signal_spec = raw_case["signal_spec"]
    if not isinstance(signal_spec, dict):
        raise ManifestValidationError(f"case {case_id!r} signal_spec must be an object")
    if not isinstance(signal_spec.get("kind"), str):
        raise ManifestValidationError(f"case {case_id!r} signal_spec.kind must be a string")

    constraints = raw_case["constraints"]
    if not isinstance(constraints, dict):
        raise ManifestValidationError(f"case {case_id!r} constraints must be an object")

    stopping = raw_case["stopping"]
    if not isinstance(stopping, dict):
        raise ManifestValidationError(f"case {case_id!r} stopping must be an object")
    if not (("n_bkps" in stopping) ^ ("pen" in stopping)):
        raise ManifestValidationError(
            f"case {case_id!r} stopping must include exactly one of n_bkps or pen"
        )

    detectors = tuple(raw_case["enabled_detectors"])
    if not detectors:
        raise ManifestValidationError(f"case {case_id!r} enabled_detectors must be non-empty")
    for detector in detectors:
        if detector not in _ALLOWED_DETECTORS:
            raise ManifestValidationError(
                f"case {case_id!r} has invalid detector {detector!r}"
            )

    models = tuple(raw_case["enabled_models"])
    if not models:
        raise ManifestValidationError(f"case {case_id!r} enabled_models must be non-empty")
    for model in models:
        if model not in _ALLOWED_MODELS:
            raise ManifestValidationError(f"case {case_id!r} has invalid model {model!r}")

    tolerance = raw_case["tolerance"]
    if not isinstance(tolerance, dict):
        raise ManifestValidationError(f"case {case_id!r} tolerance must be an object")
    tolerance_index = tolerance.get("index")
    if not isinstance(tolerance_index, int) or tolerance_index < 0:
        raise ManifestValidationError(
            f"case {case_id!r} tolerance.index must be an integer >= 0"
        )

    profile_tags = tuple(raw_case["profile_tags"])
    if not profile_tags:
        raise ManifestValidationError(f"case {case_id!r} profile_tags must be non-empty")
    unknown_profiles = sorted(set(profile_tags) - _ALLOWED_PROFILES)
    if unknown_profiles:
        raise ManifestValidationError(
            f"case {case_id!r} has unsupported profile tags: {', '.join(unknown_profiles)}"
        )

    category = raw_case["category"]
    source = raw_case["source"]
    if not isinstance(category, str) or not category:
        raise ManifestValidationError(f"case {case_id!r} category must be a non-empty string")
    if not isinstance(source, str) or not source:
        raise ManifestValidationError(f"case {case_id!r} source must be a non-empty string")

    return ParityCase(
        id=case_id,
        category=category,
        source=source,
        expected_behavior=expected_behavior,
        signal_spec=signal_spec,
        constraints=constraints,
        stopping=stopping,
        enabled_detectors=detectors,
        enabled_models=models,
        tolerance_index=tolerance_index,
        profile_tags=profile_tags,
    )


def category_counts(cases: list[ParityCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    return counts


def select_cases(cases: list[ParityCase], profile: str) -> list[ParityCase]:
    if profile not in _ALLOWED_PROFILES:
        raise ValueError(f"unsupported parity profile {profile!r}")
    return [case for case in cases if profile in case.profile_tags]


def generate_signal(case: ParityCase, manifest_path: Path) -> np.ndarray:
    spec = case.signal_spec
    kind = spec["kind"]

    if kind == "piecewise_constant":
        values = _piecewise_constant_signal(spec)
    elif kind == "volatility_shift":
        values = _volatility_shift_signal(spec)
    elif kind == "ar1_piecewise_mean":
        values = _ar1_piecewise_signal(spec)
    elif kind == "piecewise_linear":
        values = _piecewise_linear_signal(spec)
    elif kind == "real_world_file":
        values = _real_world_file_signal(spec, manifest_path=manifest_path)
    else:
        raise ValueError(f"unsupported signal_spec.kind {kind!r} for case {case.id}")

    if not np.isfinite(values[~np.isnan(values)]).all():
        raise ValueError(f"case {case.id} generated non-finite non-NaN values")

    return values.astype(np.float64, copy=False)


def deterministic_missing_preprocess(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    if not np.isnan(out).any():
        return out

    # Forward fill interior gaps from the nearest prior observation.
    for idx in range(1, out.shape[0]):
        if math.isnan(out[idx]):
            out[idx] = out[idx - 1]

    # Backfill leading gaps from the nearest following observation.
    for idx in range(out.shape[0] - 2, -1, -1):
        if math.isnan(out[idx]):
            out[idx] = out[idx + 1]

    # Degenerate all-NaN fallback.
    out[np.isnan(out)] = 0.0
    return out


def _piecewise_constant_signal(spec: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    segments = spec.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("piecewise_constant requires non-empty segments")

    values = np.concatenate(
        [
            np.full(int(segment["length"]), float(segment["mean"]), dtype=np.float64)
            for segment in segments
        ]
    )

    noise = spec.get("noise", {"distribution": "normal", "std": 0.0})
    values = values + _sample_noise(noise, rng=rng, n=values.shape[0])

    missing_gaps = spec.get("missing_gaps", [])
    for gap in missing_gaps:
        start, end = int(gap[0]), int(gap[1])
        clipped_start = max(0, min(values.shape[0], start))
        clipped_end = max(clipped_start, min(values.shape[0], end))
        values[clipped_start:clipped_end] = np.nan

    return values


def _volatility_shift_signal(spec: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    segments = spec.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("volatility_shift requires non-empty segments")

    parts: list[np.ndarray] = []
    for segment in segments:
        length = int(segment["length"])
        mean = float(segment.get("mean", 0.0))
        std = float(segment["std"])
        parts.append(rng.normal(loc=mean, scale=std, size=length))
    return np.concatenate(parts).astype(np.float64)


def _ar1_piecewise_signal(spec: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    phi = float(spec.get("phi", 0.4))
    sigma = float(spec.get("sigma", 1.0))
    segments = spec.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("ar1_piecewise_mean requires non-empty segments")

    means = np.concatenate(
        [
            np.full(int(segment["length"]), float(segment["mean"]), dtype=np.float64)
            for segment in segments
        ]
    )

    out = np.empty_like(means)
    out[0] = means[0] + rng.normal(scale=sigma)
    prev_mean = means[0]
    for idx in range(1, means.shape[0]):
        mean = means[idx]
        innovation = rng.normal(scale=sigma)
        out[idx] = mean + phi * (out[idx - 1] - prev_mean) + innovation
        prev_mean = mean
    return out


def _piecewise_linear_signal(spec: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    segments = spec.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("piecewise_linear requires non-empty segments")

    parts: list[np.ndarray] = []
    for segment in segments:
        length = int(segment["length"])
        slope = float(segment["slope"])
        intercept = float(segment["intercept"])
        index = np.arange(length, dtype=np.float64)
        parts.append(intercept + slope * index)

    values = np.concatenate(parts)
    noise = spec.get("noise", {"distribution": "normal", "std": 0.0})
    return values + _sample_noise(noise, rng=rng, n=values.shape[0])


def _real_world_file_signal(spec: dict[str, Any], manifest_path: Path) -> np.ndarray:
    fixture = spec.get("fixture")
    if not isinstance(fixture, str) or not fixture:
        raise ValueError("real_world_file requires non-empty fixture path")

    fixture_path = manifest_path.parent / fixture
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    values = payload.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError(f"real-world fixture {fixture_path} missing non-empty values")

    return np.asarray(values, dtype=np.float64)


def _sample_noise(noise_spec: dict[str, Any], rng: np.random.Generator, n: int) -> np.ndarray:
    distribution = str(noise_spec.get("distribution", "normal")).lower()
    std = float(noise_spec.get("std", 0.0))
    if distribution == "normal":
        noise = rng.normal(loc=0.0, scale=std, size=n)
    elif distribution == "student_t":
        df = float(noise_spec.get("df", 3.0))
        noise = rng.standard_t(df=df, size=n) * std
    else:
        raise ValueError(f"unsupported noise distribution {distribution!r}")

    outlier_fraction = float(noise_spec.get("outlier_fraction", 0.0))
    if outlier_fraction > 0.0:
        outlier_scale = float(noise_spec.get("outlier_scale", 8.0))
        count = int(round(n * outlier_fraction))
        if count > 0:
            indices = rng.choice(n, size=count, replace=False)
            noise[indices] += rng.normal(loc=0.0, scale=std * outlier_scale, size=count)

    return noise.astype(np.float64)


def run_parity_suite(cases: list[ParityCase], manifest_path: Path) -> list[ParityResult]:
    results: list[ParityResult] = []
    for case in cases:
        raw_values = generate_signal(case, manifest_path=manifest_path)
        values = deterministic_missing_preprocess(raw_values)

        for detector in case.enabled_detectors:
            for model in case.enabled_models:
                cpd_breakpoints, cpd_runtime_ms = run_cpd_case(
                    values=values,
                    detector=detector,
                    model=model,
                    constraints=case.constraints,
                    stopping=case.stopping,
                )
                ruptures_breakpoints, ruptures_runtime_ms = run_ruptures_case(
                    values=values,
                    detector=detector,
                    model=model,
                    constraints=case.constraints,
                    stopping=case.stopping,
                )

                cpd_change_points = tuple(_change_points(cpd_breakpoints))
                ruptures_change_points = tuple(_change_points(ruptures_breakpoints))
                exact_match = cpd_change_points == ruptures_change_points
                tolerant_match, tolerant_matches, tolerant_jaccard = compare_change_points(
                    cpd_change_points,
                    ruptures_change_points,
                    tolerance=case.tolerance_index,
                )

                cpd_cost = evaluate_segmentation_cost(values, cpd_breakpoints, model=model)
                ruptures_cost = evaluate_segmentation_cost(
                    values, ruptures_breakpoints, model=model
                )
                cost_rel_error = _relative_error(cpd_cost, ruptures_cost)

                results.append(
                    ParityResult(
                        case_id=case.id,
                        category=case.category,
                        detector=detector,
                        model=model,
                        tolerance_index=case.tolerance_index,
                        cpd_breakpoints=tuple(int(bp) for bp in cpd_breakpoints),
                        ruptures_breakpoints=tuple(int(bp) for bp in ruptures_breakpoints),
                        cpd_change_points=cpd_change_points,
                        ruptures_change_points=ruptures_change_points,
                        exact_match=exact_match,
                        tolerant_match=tolerant_match,
                        tolerant_matches=tolerant_matches,
                        tolerant_jaccard=tolerant_jaccard,
                        cpd_cost=cpd_cost,
                        ruptures_cost=ruptures_cost,
                        cost_rel_error=cost_rel_error,
                        cpd_runtime_ms=cpd_runtime_ms,
                        ruptures_runtime_ms=ruptures_runtime_ms,
                    )
                )

    return results


def run_cpd_case(
    values: np.ndarray,
    detector: str,
    model: str,
    constraints: dict[str, Any],
    stopping: dict[str, Any],
) -> tuple[list[int], float]:
    cpd = importlib.import_module("cpd")

    kwargs: dict[str, Any] = {
        "model": model,
        "min_segment_len": int(constraints.get("min_segment_len", 2)),
        "jump": int(constraints.get("jump", 1)),
    }
    max_change_points = constraints.get("max_change_points")
    if max_change_points is not None:
        kwargs["max_change_points"] = int(max_change_points)
    if detector == "binseg" and constraints.get("max_depth") is not None:
        kwargs["max_depth"] = int(constraints["max_depth"])

    detector_cls = cpd.Pelt if detector == "pelt" else cpd.Binseg
    start = time.perf_counter()
    fit = detector_cls(**kwargs).fit(values)
    if "n_bkps" in stopping:
        target_k = int(stopping["n_bkps"])
        if detector == "pelt":
            breakpoints = _cpd_pelt_predict_known_k(
                predictor=fit.predict,
                target_k=target_k,
                n_samples=values.shape[0],
            )
        else:
            result = fit.predict(n_bkps=target_k)
            breakpoints = [int(bp) for bp in result.breakpoints]
    else:
        result = fit.predict(pen=float(stopping["pen"]))
        breakpoints = [int(bp) for bp in result.breakpoints]
    runtime_ms = (time.perf_counter() - start) * 1000.0

    return breakpoints, runtime_ms


def run_ruptures_case(
    values: np.ndarray,
    detector: str,
    model: str,
    constraints: dict[str, Any],
    stopping: dict[str, Any],
) -> tuple[list[int], float]:
    rpt = importlib.import_module("ruptures")

    min_size = int(constraints.get("min_segment_len", 2))
    jump = int(constraints.get("jump", 1))

    if detector == "pelt":
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump)
    elif detector == "binseg":
        algo = rpt.Binseg(model=model, min_size=min_size, jump=jump)
    else:
        raise ValueError(f"unsupported detector {detector!r}")

    start = time.perf_counter()
    fit = algo.fit(values)
    if "n_bkps" in stopping:
        target_k = int(stopping["n_bkps"])
        if detector == "pelt":
            breakpoints = _ruptures_pelt_predict_known_k(
                predictor=fit.predict,
                target_k=target_k,
                n_samples=values.shape[0],
            )
        else:
            breakpoints = fit.predict(n_bkps=target_k)
    else:
        breakpoints = fit.predict(pen=float(stopping["pen"]))
    runtime_ms = (time.perf_counter() - start) * 1000.0

    return [int(bp) for bp in breakpoints], runtime_ms


def _ruptures_pelt_predict_known_k(
    predictor: Any,
    target_k: int,
    n_samples: int,
) -> list[int]:
    cache: dict[float, list[int]] = {}

    def _predict_for_pen(penalty: float) -> list[int]:
        cached = cache.get(penalty)
        if cached is not None:
            return cached
        predicted = [int(bp) for bp in predictor(pen=penalty)]
        cache[penalty] = predicted
        return predicted

    return _predict_known_k_via_penalty_path(
        predictor=_predict_for_pen, target_k=target_k, n_samples=n_samples
    )


def _cpd_pelt_predict_known_k(
    predictor: Any,
    target_k: int,
    n_samples: int,
) -> list[int]:
    cache: dict[float, list[int]] = {}

    def _predict_for_pen(penalty: float) -> list[int]:
        cached = cache.get(penalty)
        if cached is not None:
            return cached

        result = predictor(pen=penalty)
        predicted = [int(bp) for bp in result.breakpoints]
        cache[penalty] = predicted
        return predicted

    return _predict_known_k_via_penalty_path(
        predictor=_predict_for_pen, target_k=target_k, n_samples=n_samples
    )


def _predict_known_k_via_penalty_path(
    predictor: Callable[[float], list[int]],
    target_k: int,
    n_samples: int,
) -> list[int]:
    if target_k < 0:
        raise ValueError(f"target_k must be >= 0; got {target_k}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be >= 1; got {n_samples}")

    def _n_changes(predicted: list[int]) -> int:
        if not predicted:
            return 0
        return max(0, len(predicted) - 1)

    if target_k == 0:
        return predictor(1.0e12)

    low_pen = 1.0e-12
    low_pred = predictor(low_pen)
    low_changes = _n_changes(low_pred)
    if low_changes < target_k:
        return low_pred

    high_pen = 1.0
    high_pred = predictor(high_pen)
    high_changes = _n_changes(high_pred)
    for _ in range(80):
        if high_changes <= target_k:
            break
        high_pen *= 2.0
        high_pred = predictor(high_pen)
        high_changes = _n_changes(high_pred)

    best_pred = low_pred
    best_delta = abs(low_changes - target_k)
    if abs(high_changes - target_k) < best_delta:
        best_pred = high_pred
        best_delta = abs(high_changes - target_k)

    if low_changes == target_k:
        return low_pred
    if high_changes == target_k:
        return high_pred

    if high_changes > target_k:
        return high_pred if abs(high_changes - target_k) <= best_delta else best_pred

    left_pen = low_pen
    right_pen = high_pen
    for _ in range(64):
        mid_pen = 0.5 * (left_pen + right_pen)
        mid_pred = predictor(mid_pen)
        mid_changes = _n_changes(mid_pred)
        delta = abs(mid_changes - target_k)
        if delta < best_delta:
            best_pred = mid_pred
            best_delta = delta
        if mid_changes == target_k:
            return mid_pred
        if mid_changes > target_k:
            left_pen = mid_pen
        else:
            right_pen = mid_pen

    return best_pred


def _change_points(breakpoints: list[int]) -> list[int]:
    n = breakpoints[-1]
    return [bp for bp in breakpoints if bp < n]


def compare_change_points(
    left: tuple[int, ...] | list[int],
    right: tuple[int, ...] | list[int],
    tolerance: int,
) -> tuple[bool, int, float]:
    left_sorted = tuple(sorted(int(x) for x in left))
    right_sorted = tuple(sorted(int(x) for x in right))

    i = 0
    j = 0
    matches = 0
    while i < len(left_sorted) and j < len(right_sorted):
        delta = left_sorted[i] - right_sorted[j]
        if abs(delta) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif delta < -tolerance:
            i += 1
        else:
            j += 1

    union = len(left_sorted) + len(right_sorted) - matches
    if union == 0:
        jaccard = 1.0
    else:
        jaccard = matches / union

    tolerant_match = matches == len(left_sorted) == len(right_sorted)
    return tolerant_match, matches, jaccard


def evaluate_segmentation_cost(values: np.ndarray, breakpoints: list[int], model: str) -> float:
    x = np.asarray(values, dtype=np.float64)
    if breakpoints != sorted(set(breakpoints)):
        raise ValueError(f"breakpoints must be strictly increasing and unique: {breakpoints}")
    if not breakpoints:
        raise ValueError("breakpoints must be non-empty")
    if breakpoints[-1] != x.shape[0]:
        raise ValueError(
            f"breakpoints must include n as final element; got {breakpoints[-1]}, expected {x.shape[0]}"
        )

    total = 0.0
    start = 0
    for end in breakpoints:
        if end <= start:
            raise ValueError(f"invalid segment boundaries: [{start}, {end})")
        segment = x[start:end]
        m = float(segment.shape[0])

        if model == "l2":
            mean = float(segment.mean())
            total += float(np.square(segment - mean).sum())
        elif model == "normal":
            mean = float(segment.mean())
            raw_var = float(np.square(segment - mean).mean())
            variance = max(raw_var, _VAR_FLOOR)
            total += m * math.log(variance)
        else:
            raise ValueError(f"unsupported model {model!r}")

        start = end

    return float(total)


def _relative_error(left: float, right: float) -> float:
    denom = max(abs(right), 1e-12)
    return abs(left - right) / denom


def summarize_results(results: list[ParityResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}

    for result in results:
        key = f"{result.detector}:{result.model}"
        bucket = grouped.setdefault(
            key,
            {
                "total": 0.0,
                "exact_pass": 0.0,
                "tolerant_pass": 0.0,
                "cost_exact_checked": 0.0,
                "cost_exact_pass": 0.0,
                "mean_jaccard": 0.0,
            },
        )
        bucket["total"] += 1.0
        bucket["exact_pass"] += 1.0 if result.exact_match else 0.0
        bucket["tolerant_pass"] += 1.0 if result.tolerant_match else 0.0
        bucket["mean_jaccard"] += result.tolerant_jaccard
        if result.exact_match:
            bucket["cost_exact_checked"] += 1.0
            bucket["cost_exact_pass"] += 1.0 if result.cost_rel_error <= 1e-6 else 0.0

    for bucket in grouped.values():
        total = max(bucket["total"], 1.0)
        bucket["exact_rate"] = bucket["exact_pass"] / total
        bucket["tolerant_rate"] = bucket["tolerant_pass"] / total
        bucket["mean_jaccard"] = bucket["mean_jaccard"] / total
        checked = max(bucket["cost_exact_checked"], 1.0)
        bucket["cost_exact_rate"] = bucket["cost_exact_pass"] / checked

    return grouped


def results_to_jsonable(results: list[ParityResult]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": result.case_id,
            "category": result.category,
            "detector": result.detector,
            "model": result.model,
            "tolerance_index": result.tolerance_index,
            "cpd_breakpoints": list(result.cpd_breakpoints),
            "ruptures_breakpoints": list(result.ruptures_breakpoints),
            "cpd_change_points": list(result.cpd_change_points),
            "ruptures_change_points": list(result.ruptures_change_points),
            "exact_match": result.exact_match,
            "tolerant_match": result.tolerant_match,
            "tolerant_matches": result.tolerant_matches,
            "tolerant_jaccard": result.tolerant_jaccard,
            "cpd_cost": result.cpd_cost,
            "ruptures_cost": result.ruptures_cost,
            "cost_rel_error": result.cost_rel_error,
            "cpd_runtime_ms": result.cpd_runtime_ms,
            "ruptures_runtime_ms": result.ruptures_runtime_ms,
        }
        for result in results
    ]
