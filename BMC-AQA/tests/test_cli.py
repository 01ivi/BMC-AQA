"""CPU train/save/load/evaluate integration using temporary FS1000-like data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from smoke_test import create_tiny_fs1000


class CliTests(unittest.TestCase):
    def test_training_and_evaluation_preserve_split_and_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="bmc-cli-") as temporary:
            root = Path(temporary)
            create_tiny_fs1000(root / "data")
            config = (ROOT / "configs/fs1000.toml").read_text()
            replacements = {"clip_num = 95": "clip_num = 3", "hidden_dim = 256": "hidden_dim = 8",
                            "heads = 4": "heads = 2", "layers = 2": "layers = 1",
                            "key_dim = 128": "key_dim = 4", "epochs = 100": "epochs = 1",
                            "batch_size = 32": "batch_size = 2", "dropout = 0.1": "dropout = 0.0",
                            "val_fraction = 0.15": "val_fraction = 0.25"}
            for old, new in replacements.items():
                config = config.replace(old, new)
            (root / "tiny.toml").write_text(config)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

            def run(script, *arguments):
                return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)],
                                      cwd=ROOT, capture_output=True, text=True, env=env, timeout=60)

            trained = run("train.py", "--config", root / "tiny.toml", "--data-root", root / "data",
                          "--output-dir", root / "train", "--device", "cpu")
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
            checkpoint = root / "train/best_mse.pt"
            self.assertTrue(checkpoint.exists())
            payload = torch.load(checkpoint, weights_only=True)
            self.assertFalse(set(payload["memory_ids"]) & set(payload["validation_ids"]))
            self.assertEqual(len(payload["memory_ids"]), 6)
            for protocol in ("all-imr", "fixed"):
                evaluated = run("evaluate.py", "--checkpoint", checkpoint, "--data-root", root / "data",
                                "--output-dir", root / protocol, "--device", "cpu", "--protocol", protocol)
                self.assertEqual(evaluated.returncode, 0, evaluated.stdout + evaluated.stderr)
                rows = json.loads((root / protocol / "metrics.json").read_text())
                self.assertEqual(len(rows), 7 if protocol == "all-imr" else 8)
                self.assertTrue(all(row["mse"] >= 0 for row in rows))
                self.assertTrue((root / protocol / "predictions.csv").is_file())
            wrong_memory = root / "wrong.pt"
            memory = torch.load(root / "train/memory.pt", weights_only=True)
            memory["ids"] = list(reversed(memory["ids"]))
            torch.save(memory, wrong_memory)
            rejected = run("evaluate.py", "--checkpoint", checkpoint, "--memory-path", wrong_memory,
                           "--data-root", root / "data", "--output-dir", root / "wrong", "--device", "cpu")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Memory identifiers differ", rejected.stderr)


if __name__ == "__main__":
    unittest.main()

