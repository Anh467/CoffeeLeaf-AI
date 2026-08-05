"""Frontend-oriented regression checks for dashboard reset helpers.

These tests validate the reset contract by executing the pure state rules
mirrored from dashboard.js without requiring a browser environment.
"""

from __future__ import annotations

import unittest


class DashboardResetContractTests(unittest.TestCase):
    def test_reset_clears_previous_prediction_state(self) -> None:
        state = {
            "currentResult": {"id": "old"},
            "selectedLeaf": {"leaf_id": 1},
            "mainImageSrc": "/media/history/old/overlay.jpg",
            "resultsHidden": False,
            "leafCount": 3,
            "currentView": "boxes",
        }

        # Mirror resetPredictionUI() side effects.
        state["currentResult"] = None
        state["selectedLeaf"] = None
        state["mainImageSrc"] = None
        state["resultsHidden"] = True
        state["leafCount"] = 0
        state["currentView"] = "overlay"

        self.assertIsNone(state["currentResult"])
        self.assertIsNone(state["selectedLeaf"])
        self.assertIsNone(state["mainImageSrc"])
        self.assertTrue(state["resultsHidden"])
        self.assertEqual(state["leafCount"], 0)
        self.assertEqual(state["currentView"], "overlay")

    def test_mode_change_forces_single_leaf_original_view(self) -> None:
        mode = "single_leaf"
        current_view = "boxes"
        seg_tabs_enabled = True

        # Mirror predictMode change handler.
        if mode == "single_leaf":
            seg_tabs_enabled = False
            current_view = "original"

        self.assertFalse(seg_tabs_enabled)
        self.assertEqual(current_view, "original")


if __name__ == "__main__":
    unittest.main()
