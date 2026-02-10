import copy
import unittest
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import schema_gate  # noqa: E402


class SchemaGateTests(unittest.TestCase):
    @staticmethod
    def _valid_config_fixture() -> dict:
        return {
            "schema_version": 0,
            "kind": "pipeline_spec",
            "payload": {
                "preprocess": {
                    "detrend": {"method": "polynomial", "degree": 2},
                    "deseasonalize": {"method": "stl_like", "period": 4},
                    "winsorize": {"lower_quantile": 0.05, "upper_quantile": 0.95},
                    "robust_scale": {"mad_epsilon": 1.0e-9, "normal_consistency": 1.4826},
                }
            },
        }

    @staticmethod
    def _load_config_schema() -> dict:
        schema_path = (
            schema_gate.CPD_ROOT / "schemas" / "config" / "pipeline_spec.v0.schema.json"
        )
        return schema_gate._as_dict(schema_gate._read_json(schema_path), str(schema_path))

    def test_validate_repo_passes_for_workspace(self):
        errors = schema_gate.validate_repo(schema_gate.REPO_ROOT)
        self.assertEqual(errors, [])

    def test_config_fixture_requires_schema_version_marker(self):
        with self.assertRaisesRegex(ValueError, "schema_version"):
            schema_gate.validate_config_fixture(
                {
                    "kind": "pipeline_spec",
                    "payload": {},
                }
            )

    def test_config_fixture_rejects_wrong_schema_version(self):
        with self.assertRaisesRegex(ValueError, "schema_version must be 0"):
            schema_gate.validate_config_fixture(
                {
                    "schema_version": 1,
                    "kind": "pipeline_spec",
                    "payload": {},
                }
            )

    def test_config_schema_rejects_missing_preprocess_contract(self):
        schema = self._load_config_schema()
        broken = copy.deepcopy(schema)
        del broken["properties"]["payload"]["properties"]["preprocess"]
        with self.assertRaisesRegex(ValueError, "preprocess"):
            schema_gate.validate_config_schema(broken)

    def test_config_fixture_accepts_valid_preprocess_contract(self):
        schema_gate.validate_config_fixture(self._valid_config_fixture())

    def test_config_fixture_rejects_unknown_preprocess_stage(self):
        fixture = self._valid_config_fixture()
        fixture["payload"]["preprocess"]["unknown_stage"] = {}
        with self.assertRaisesRegex(ValueError, "unsupported preprocess keys"):
            schema_gate.validate_config_fixture(fixture)

    def test_config_fixture_rejects_invalid_detrend_method(self):
        fixture = self._valid_config_fixture()
        fixture["payload"]["preprocess"]["detrend"] = {"method": "cubic"}
        with self.assertRaisesRegex(ValueError, "detrend\\.method"):
            schema_gate.validate_config_fixture(fixture)

    def test_config_fixture_rejects_polynomial_without_degree(self):
        fixture = self._valid_config_fixture()
        fixture["payload"]["preprocess"]["detrend"] = {"method": "polynomial"}
        with self.assertRaisesRegex(ValueError, "degree >= 1"):
            schema_gate.validate_config_fixture(fixture)

    def test_config_fixture_rejects_stl_like_period_below_two(self):
        fixture = self._valid_config_fixture()
        fixture["payload"]["preprocess"]["deseasonalize"] = {
            "method": "stl_like",
            "period": 1,
        }
        with self.assertRaisesRegex(ValueError, ">= 2"):
            schema_gate.validate_config_fixture(fixture)

    def test_config_fixture_rejects_invalid_winsorize_quantiles(self):
        fixture = self._valid_config_fixture()
        fixture["payload"]["preprocess"]["winsorize"] = {
            "lower_quantile": 0.8,
            "upper_quantile": 0.2,
        }
        with self.assertRaisesRegex(ValueError, "lower_quantile < upper_quantile"):
            schema_gate.validate_config_fixture(fixture)

    def test_config_fixture_rejects_nonpositive_robust_scale(self):
        fixture = self._valid_config_fixture()
        fixture["payload"]["preprocess"]["robust_scale"] = {
            "mad_epsilon": 0.0,
            "normal_consistency": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            schema_gate.validate_config_fixture(fixture)

    def test_checkpoint_fixture_rejects_bad_crc(self):
        with self.assertRaisesRegex(ValueError, "payload_crc32"):
            schema_gate.validate_checkpoint_fixture(
                {
                    "schema_version": 0,
                    "detector_id": "bocpd",
                    "engine_version": "0.1.0",
                    "created_at_ns": 1,
                    "payload_codec": "json",
                    "payload_crc32": "DEADBEEF",
                    "payload": {},
                }
            )


if __name__ == "__main__":
    unittest.main()
