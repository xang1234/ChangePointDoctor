"""Compatibility import alias for changepoint-doctor.

Canonical Python import is `cpd`; this module exists so users who try
`import changepoint_doctor` still reach the same API surface.
"""

from cpd import (
    Binseg,
    Bocpd,
    BuildInfo,
    Cusum,
    Diagnostics,
    Fpop,
    OfflineChangePointResult,
    OnlineStepResult,
    Pelt,
    PageHinkley,
    PruningStats,
    SegmentStats,
    SmokeDetector,
    __version__,
    detect_offline,
    smoke_detect,
)

__all__ = [
    "__version__",
    "PruningStats",
    "BuildInfo",
    "SegmentStats",
    "Diagnostics",
    "OfflineChangePointResult",
    "OnlineStepResult",
    "Pelt",
    "Binseg",
    "Fpop",
    "Bocpd",
    "Cusum",
    "PageHinkley",
    "detect_offline",
    "SmokeDetector",
    "smoke_detect",
]
