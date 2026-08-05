"""Unit tests for inspect_psc_candidate helpers (no live PSC HTTP calls)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "inspect_psc_candidate.py"


def _load_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("inspect_psc_candidate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


inspect_psc = _load_module()


def _valid_payload() -> dict:
    return {
        "profile": [0.1, 0.5, 1.0, 0.2],
        "FvP": [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        "TvP": [[0.0, 0.1, 0.0], [0.2, 0.9, 0.1]],
        "DMc": [0.0, 0.4, 1.0, 0.3],
        "metadata": {
            "plot_id": 11732926,
            "survey": "AO327",
            "dm": 42.1,
            "period": 0.0123,
            "snr": 8.5,
            "unused_extra": "ok",
        },
    }


class TestInspectPscHelpers(unittest.TestCase):
    def test_build_plot_url(self) -> None:
        url = inspect_psc.build_plot_url(11732926)
        self.assertEqual(
            url,
            "https://pulsars.nanograv.org/api/psrsearch/plots/11732926/json",
        )

    def test_extract_metadata_nested_and_useful_keys(self) -> None:
        payload = _valid_payload()
        report = inspect_psc.inspect_candidate_payload(payload)

        self.assertIn("profile", report["top_level_keys"])
        self.assertIn("unused_extra", report["metadata_keys"])
        self.assertEqual(report["useful_metadata"]["survey"], "AO327")
        self.assertEqual(report["useful_metadata"]["dm"], 42.1)
        self.assertNotIn("unused_extra", report["useful_metadata"])

        self.assertEqual(report["arrays"]["profile"]["shape"], (4,))
        self.assertEqual(report["arrays"]["FvP"]["shape"], (3, 2))
        self.assertEqual(report["arrays"]["TvP"]["shape"], (2, 3))
        self.assertEqual(report["arrays"]["DMc"]["shape"], (4,))
        for name in inspect_psc.ARRAY_KEYS:
            self.assertEqual(report["arrays"][name]["issues"], [])

    def test_detects_missing_nan_inf_ragged_and_bad_ndim(self) -> None:
        payload = {
            "profile": None,
            "FvP": [[1.0, 2.0], [3.0]],  # ragged
            "TvP": [1.0, 2.0, 3.0],  # unexpected 1-D
            "DMc": [0.0, float("nan"), float("inf")],
            "metadata": {},
        }
        report = inspect_psc.inspect_candidate_payload(payload)

        self.assertIn("missing", report["arrays"]["profile"]["issues"])
        self.assertIn("ragged", report["arrays"]["FvP"]["issues"])
        self.assertTrue(
            any(
                issue.startswith("unexpected_ndim")
                for issue in report["arrays"]["TvP"]["issues"]
            )
        )
        self.assertIn("nan", report["arrays"]["DMc"]["issues"])
        self.assertIn("inf", report["arrays"]["DMc"]["issues"])

    def test_to_numpy_array_and_ragged_helper(self) -> None:
        array = inspect_psc.to_numpy_array([1, 2, 3], name="profile")
        np.testing.assert_array_equal(array, np.array([1.0, 2.0, 3.0]))
        self.assertTrue(inspect_psc.is_ragged_sequence([[1], [1, 2]]))
        self.assertFalse(inspect_psc.is_ragged_sequence([[1, 2], [3, 4]]))

    def test_format_inspection_report_contains_shapes(self) -> None:
        report = inspect_psc.inspect_candidate_payload(_valid_payload())
        text = inspect_psc.format_inspection_report(report)
        self.assertIn("profile shape: (4,)", text)
        self.assertIn("FvP shape: (3, 2)", text)
        self.assertIn("survey: AO327", text)

    def test_fetch_plot_json_uses_session_and_handles_http_error(self) -> None:
        session = MagicMock()
        ok_response = MagicMock()
        ok_response.ok = True
        ok_response.json.return_value = _valid_payload()
        session.get.return_value = ok_response

        payload = inspect_psc.fetch_plot_json(11732926, session=session, timeout_s=5)
        self.assertIn("profile", payload)
        session.get.assert_called_once()
        kwargs = session.get.call_args.kwargs
        self.assertEqual(kwargs["timeout"], 5)

        bad_response = MagicMock()
        bad_response.ok = False
        bad_response.status_code = 404
        bad_response.reason = "Not Found"
        session.get.return_value = bad_response
        with self.assertRaisesRegex(RuntimeError, "PSC HTTP 404"):
            inspect_psc.fetch_plot_json(999, session=session)


if __name__ == "__main__":
    unittest.main()
