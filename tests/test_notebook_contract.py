import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class NotebookContractTests(unittest.TestCase):
    def test_notebook_exposes_minimal_mlops_contract(self) -> None:
        notebook = json.loads(
            (REPO_ROOT / "coffee_leaf_experiment.ipynb").read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )

        self.assertIn('os.getenv("DATASET_DIR"', source)
        self.assertIn('os.getenv("NUM_EPOCHS"', source)
        self.assertIn('os.getenv("MLOPS_METRICS_PATH"', source)
        self.assertIn('"schema_version": 1', source)
        self.assertIn('"models": mlops_models', source)
        self.assertIn('"candidate": max(mlops_models', source)
        self.assertIn('"model_name": res["model_name"]', source)
        self.assertIn('"best_val_accuracy"', source)
        self.assertIn('"checkpoint":', source)


if __name__ == "__main__":
    unittest.main()
