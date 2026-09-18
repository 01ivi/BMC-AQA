"""Small TOML configuration loader shared by the public command-line tools."""
from __future__ import annotations

import json
from pathlib import Path
import tomllib


def load_config(path):
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    for section in ("data", "model", "retrieval", "loss", "training"):
        if section not in config:
            raise ValueError(f"Missing [{section}] in {path}")
    return config


def save_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

