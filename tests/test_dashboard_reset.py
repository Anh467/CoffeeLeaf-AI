"""Frontend-oriented regression checks for the redesigned dashboard."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DashboardMarkupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = (ROOT / "app" / "templates" / "index.html").read_text(
            encoding="utf-8"
        )
        self.js = (ROOT / "app" / "static" / "js" / "dashboard.js").read_text(
            encoding="utf-8"
        )

    def test_detection_mode_removed_from_dom(self) -> None:
        self.assertNotIn("Detection mode", self.index)
        self.assertNotIn("predict-mode", self.index)
        self.assertNotIn("Auto detect leaves", self.index)
        self.assertNotIn("Whole image segmentation", self.index)
        self.assertNotIn('id="predict-mode"', self.index)

    def test_home_has_upload_and_disabled_predict(self) -> None:
        self.assertIn("home-state", self.index)
        self.assertIn("Bắt đầu phân tích", self.index)
        self.assertIn("disabled", self.index)
        self.assertIn('id="results"', self.index)
        self.assertIn("d-none", self.index)

    def test_request_sends_file_not_mode(self) -> None:
        self.assertIn('body.append("file", selectedFile)', self.js)
        self.assertNotIn('body.append("mode"', self.js)
        self.assertNotIn("predictMode", self.js)

    def test_metrics_modal_controls_exist(self) -> None:
        self.assertIn('id="metrics-modal"', self.index)
        self.assertIn('id="open-metrics-btn"', self.index)
        self.assertIn("Xem thông số", self.index)
        self.assertIn("Escape", self.js)
        self.assertIn("closeMetricsModal", self.js)

    def test_fallback_rendering_helpers_exist(self) -> None:
        self.assertIn("renderFullImageFallback", self.js)
        self.assertIn("Kết quả toàn ảnh", self.js)
        self.assertIn("analysis_scope", self.js)


class DashboardResetContractTests(unittest.TestCase):
    def test_reset_clears_previous_prediction_state(self) -> None:
        state = {
            "currentResult": {"id": "old"},
            "selectedLeafId": 1,
            "resultsHidden": False,
            "homeHidden": True,
        }
        state["currentResult"] = None
        state["selectedLeafId"] = None
        state["resultsHidden"] = True
        state["homeHidden"] = False
        self.assertIsNone(state["currentResult"])
        self.assertIsNone(state["selectedLeafId"])
        self.assertTrue(state["resultsHidden"])
        self.assertFalse(state["homeHidden"])


if __name__ == "__main__":
    unittest.main()
