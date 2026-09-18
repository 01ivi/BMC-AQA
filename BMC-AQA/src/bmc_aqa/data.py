from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import MODALITIES

DEFAULT_MODALITIES = MODALITIES


@dataclass(frozen=True)
class LabelEntry:
    sample_id: str
    score: float


def _as_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def _load_npy(path: Path):
    return np.load(path, allow_pickle=True)


def temporal_crop_or_pad(
    feat: np.ndarray,
    clip_num: int,
    train: bool,
    deterministic: bool = False,
    crop_start: Optional[int] = None,
) -> np.ndarray:
    """Crop/pad one temporal feature array to a fixed number of clips."""
    feat = np.asarray(feat)
    if feat.ndim == 1:
        feat = feat[:, None]
    if feat.shape[0] == clip_num:
        return feat
    if feat.shape[0] > clip_num:
        if crop_start is not None:
            start = min(max(int(crop_start), 0), feat.shape[0] - clip_num)
        elif train and not deterministic:
            start = np.random.randint(0, feat.shape[0] - clip_num + 1)
        else:
            start = (feat.shape[0] - clip_num) // 2
        return feat[start : start + clip_num]

    out = np.zeros((clip_num, feat.shape[-1]), dtype=feat.dtype)
    out[: feat.shape[0]] = feat
    return out


class AQAFeatureDataset(Dataset):
    """Feature-level loader compatible with FS1000, Fis-V, and RG.

    The class preserves the feature layout used by MCMoE:
    FS1000 stores one `.npy` file per sample and modality; Fis-V/RG store
    modality dictionaries in single `.npy` files.
    """

    def __init__(
        self,
        dataset: str,
        rgb_path: str,
        audio_path: str,
        flow_path: str,
        label_path: str,
        clip_num: int,
        action_type: str,
        train: bool = True,
        modalities: Sequence[str] = DEFAULT_MODALITIES,
        deterministic: bool = False,
    ) -> None:
        tasks = {
            "FS1000": ("TES", "PCS", "SS", "TR", "PE", "CO", "IN"),
            "FisV": ("TES", "PCS"),
            "RG": ("Ball", "Clubs", "Hoop", "Ribbon"),
        }
        if dataset not in tasks or action_type not in tasks[dataset]:
            raise ValueError(f"Unsupported dataset/task: {dataset}/{action_type}")
        if clip_num < 1:
            raise ValueError("clip_num must be positive")
        self.dataset = dataset
        self.rgb_path = _as_path(rgb_path)
        self.audio_path = _as_path(audio_path)
        self.flow_path = _as_path(flow_path)
        self.label_path = _as_path(label_path)
        self.clip_num = clip_num
        self.action_type = action_type
        self.train = train
        self.modalities = tuple(modalities)
        self.deterministic = deterministic
        self._dict_cache: Dict[Path, Dict[str, np.ndarray]] = {}

        self.score_range = self._score_range()
        self.labels = self._read_labels()
        if not self.labels:
            raise ValueError(f"No {action_type} labels found in {self.label_path}")
        for entry in self.labels:
            if not np.isfinite(entry.score) or not 0 <= entry.score <= self.score_range:
                raise ValueError(f"Invalid {action_type} score for {entry.sample_id}: {entry.score}")

    def clone(
        self,
        *,
        train: Optional[bool] = None,
        deterministic: Optional[bool] = None,
    ) -> "AQAFeatureDataset":
        return AQAFeatureDataset(
            dataset=self.dataset,
            rgb_path=str(self.rgb_path),
            audio_path=str(self.audio_path),
            flow_path=str(self.flow_path),
            label_path=str(self.label_path),
            clip_num=self.clip_num,
            action_type=self.action_type,
            train=self.train if train is None else train,
            modalities=self.modalities,
            deterministic=self.deterministic if deterministic is None else deterministic,
        )

    def _score_range(self) -> float:
        if self.dataset == "FS1000":
            ranges = {
                "TES": 130.0,
                "PCS": 60.0,
                "SS": 10.0,
                "TR": 10.0,
                "PE": 10.0,
                "CO": 10.0,
                "IN": 10.0,
            }
            return ranges[self.action_type]
        if self.dataset == "FisV":
            return 45.0 if self.action_type == "TES" else 40.0
        if self.dataset == "RG":
            return 25.0
        raise ValueError(f"Unsupported dataset: {self.dataset}")

    def _read_labels(self) -> List[LabelEntry]:
        if self.dataset == "FS1000":
            score_idx = {"TES": 1, "PCS": 2, "SS": 3, "TR": 4, "PE": 5, "CO": 6, "IN": 7}
            entries: List[LabelEntry] = []
            with self.label_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    score = float(parts[score_idx[self.action_type]])
                    if self.action_type == "PCS":
                        score = score / float(parts[8])
                    entries.append(LabelEntry(parts[0], score))
            return entries

        if self.dataset == "FisV":
            score_idx = {"TES": 1, "PCS": 2}
            entries = []
            with self.label_path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle):
                    if line_no == 0:
                        continue
                    parts = line.strip().split()
                    if parts:
                        entries.append(LabelEntry(parts[0], float(parts[score_idx[self.action_type]])))
            return entries

        if self.dataset == "RG":
            score_idx = {"Difficulty_Score": 1, "Execution_Score": 2, "Total_Score": 3}
            entries = []
            with self.label_path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle):
                    if line_no == 0:
                        continue
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if self.action_type == parts[0].split("_")[0]:
                        entries.append(LabelEntry(parts[0], float(parts[score_idx["Total_Score"]])))
            return entries

        raise ValueError(f"Unsupported dataset: {self.dataset}")

    def _dict_feature_path(self, modality: str) -> Path:
        if self.dataset == "FisV":
            names = {
                "rgb": "FISV_rgb_VST.npy",
                "audio": "FISV_audio_AST.npy",
                "flow": "FISV_flow_I3D.npy",
            }
            roots = {"rgb": self.rgb_path, "audio": self.audio_path, "flow": self.flow_path}
            return roots[modality] / names[modality]

        if self.dataset == "RG":
            suffixes = {
                "rgb": "rgb_VST.npy",
                "audio": "audio_AST.npy",
                "flow": "flow_I3D.npy",
            }
            roots = {"rgb": self.rgb_path, "audio": self.audio_path, "flow": self.flow_path}
            return roots[modality] / f"{self.action_type}_{suffixes[modality]}"

        raise ValueError(f"{self.dataset} does not use dictionary-style feature files.")

    def _load_feature(self, modality: str, sample_id: str) -> np.ndarray:
        if self.dataset == "FS1000":
            roots = {"rgb": self.rgb_path, "audio": self.audio_path, "flow": self.flow_path}
            feat = _load_npy(roots[modality] / f"{sample_id}.npy")
            if modality == "rgb" and feat.ndim == 3:
                feat = feat.mean(axis=1)
            return np.asarray(feat, dtype=np.float32)

        feature_path = self._dict_feature_path(modality)
        if feature_path not in self._dict_cache:
            loaded = _load_npy(feature_path)
            self._dict_cache[feature_path] = loaded.item()
        return np.asarray(self._dict_cache[feature_path][sample_id], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, object]:
        entry = self.labels[index]
        raw_features = {
            modality: self._load_feature(modality, entry.sample_id)
            for modality in self.modalities
        }
        # Use one crop start to preserve alignment across modality streams.
        reference_name = "rgb" if "rgb" in raw_features else self.modalities[0]
        reference = raw_features[reference_name]
        shared_crop_start: Optional[int] = None
        if reference.shape[0] > self.clip_num:
            if self.train and not self.deterministic:
                shared_crop_start = int(
                    np.random.randint(0, reference.shape[0] - self.clip_num + 1)
                )
            else:
                shared_crop_start = (reference.shape[0] - self.clip_num) // 2

        features: Dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            feat = raw_features[modality]
            feat = temporal_crop_or_pad(
                feat,
                clip_num=self.clip_num,
                train=self.train,
                deterministic=self.deterministic,
                crop_start=shared_crop_start,
            )
            features[modality] = torch.from_numpy(feat).float()

        label = float(entry.score / self.score_range)
        return {
            "id": entry.sample_id,
            "features": features,
            "label": torch.tensor(label, dtype=torch.float32),
            "raw_label": torch.tensor(entry.score, dtype=torch.float32),
        }

    def infer_input_dims(self) -> Dict[str, int]:
        if len(self) == 0:
            raise ValueError("Cannot infer input dimensions from an empty dataset.")
        sample = self.clone(train=False, deterministic=True)[0]
        features = sample["features"]
        assert isinstance(features, dict)
        return {name: int(tensor.shape[-1]) for name, tensor in features.items()}



def available_dataset_presets(root: str | os.PathLike[str]) -> Dict[str, Dict[str, str]]:
    """Paths for the public feature layouts (see docs/data.md)."""
    root_path = _as_path(root)
    return {
        "FS1000": {
            "rgb_path": str(root_path / "FS1000" / "output_feature_fs1000_new"),
            "audio_path": str(root_path / "FS1000" / "ast_feature_fs1000_new"),
            "flow_path": str(root_path / "FS1000" / "i3d_avg_clip8_5s_fs1000"),
            "train_label_path": str(root_path / "FS1000" / "train_fs1000_new.txt"),
            "test_label_path": str(root_path / "FS1000" / "val_fs1000_new.txt"),
        },
        "FisV": {
            "rgb_path": str(root_path / "Fis-V" / "Fis-feature"),
            "audio_path": str(root_path / "Fis-V" / "Fis-feature"),
            "flow_path": str(root_path / "Fis-V" / "Fis-feature"),
            "train_label_path": str(root_path / "Fis-V" / "train.txt"),
            "test_label_path": str(root_path / "Fis-V" / "test.txt"),
        },
        "RG": {
            "rgb_path": str(root_path / "RG" / "RG-feature"),
            "audio_path": str(root_path / "RG" / "RG-feature"),
            "flow_path": str(root_path / "RG" / "RG-feature"),
            "train_label_path": str(root_path / "RG" / "train.txt"),
            "test_label_path": str(root_path / "RG" / "test.txt"),
        },
    }
