from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmc_aqa.constants import MODALITIES
from bmc_aqa.data import AQAFeatureDataset


class DatasetTests(unittest.TestCase):
    def test_fisv_and_rg_dictionary_features(self):
        with tempfile.TemporaryDirectory(prefix="bmc-data-") as temporary:
            root = Path(temporary)
            for dataset, task, prefix, sample_id, header, row, expected in (
                ("FisV", "TES", "FISV", "sample", "id TES PCS", "sample 22.5 20", 0.5),
                ("RG", "Ball", "Ball", "Ball_001", "id Difficulty Execution Total", "Ball_001 5 10 15", 0.6),
            ):
                feature_dir = root / dataset
                feature_dir.mkdir()
                for name, suffix in zip(MODALITIES, ("rgb_VST", "flow_I3D", "audio_AST")):
                    np.save(feature_dir / f"{prefix}_{suffix}.npy", {sample_id: np.ones((2, 4), dtype=np.float32)})
                label_path = feature_dir / "labels.txt"
                label_path.write_text(header + "\n" + row + "\n")
                source = AQAFeatureDataset(dataset, str(feature_dir), str(feature_dir), str(feature_dir),
                                           str(label_path), 3, task, train=False)
                sample = source[0]
                self.assertAlmostEqual(float(sample["label"]), expected, places=6)
                self.assertEqual(tuple(sample["features"]), MODALITIES)
                for value in sample["features"].values():
                    self.assertEqual(value.shape, (3, 4))
                    torch.testing.assert_close(value[-1], torch.zeros(4))

    def test_fs1000_view_pooling_pcs_factor_and_shared_crop(self):
        with tempfile.TemporaryDirectory(prefix="bmc-data-") as temporary:
            root = Path(temporary)
            temporal = np.arange(6, dtype=np.float32)[:, None].repeat(4, axis=1)
            for name in MODALITIES:
                path = root / name
                path.mkdir()
                value = np.stack((temporal, temporal), 1) if name == "rgb" else temporal
                np.save(path / "sample.npy", value)
            labels = root / "labels.txt"
            labels.write_text("sample 65 30 5 5 5 5 5 2\n")
            source = AQAFeatureDataset("FS1000", str(root / "rgb"), str(root / "audio"), str(root / "flow"),
                                       str(labels), 2, "PCS", train=False)
            sample = source[0]
            self.assertEqual(float(sample["label"]), 0.25)
            self.assertEqual(float(sample["raw_label"]), 15)
            for name in MODALITIES:
                torch.testing.assert_close(sample["features"][name], torch.from_numpy(temporal[2:4]))


if __name__ == "__main__":
    unittest.main()

