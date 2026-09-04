# ============================================================
# Script:  config_loader.py
# Release: 1.0
# Version: v1.00
# Purpose: Load and validate config.json; provide typed access helpers
# ============================================================

import json
from pathlib import Path


def load_config(config_path: str | Path | None = None) -> dict:
    """Load config.json. Defaults to repo root config.json relative to this file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def get_jsonl_path(cfg: dict, lang: str) -> Path:
    root = Path(cfg["paths"]["data_root"])
    pattern = cfg["paths"]["jsonl_pattern"]
    return root / pattern.format(lang=lang)


def get_audio_root(cfg: dict, lang: str) -> Path:
    return Path(cfg["paths"]["audio_root"][lang])


def get_intermediate_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["intermediate_dir"])


def get_results_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["results_dir"])


def get_field(cfg: dict, internal_name: str) -> str:
    """Resolve internal field name to actual JSONL field name from config."""
    return cfg["jsonl_fields"][internal_name]
