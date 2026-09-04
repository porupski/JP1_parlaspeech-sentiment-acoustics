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


def build_audio_index(audio_root: Path) -> dict:
    """Recursively index all audio files under audio_root.
    Returns {filename: absolute_path_str}.
    Needed because JSONL audio paths may not match the local directory layout
    (e.g. v4 JSONL referencing files stored under v3 part-folder structure).
    """
    index = {}
    for ext in ("*.flac", "*.wav", "*.mp3"):
        for p in audio_root.rglob(ext):
            index[p.name] = str(p)
    return index


def resolve_audio_path(audio_rel: str,
                       audio_root: Path,
                       index: dict) -> "Path | None":
    """Resolve audio path with fallback:
    1. Absolute path in audio_rel — use directly.
    2. audio_root / audio_rel — check exists.
    3. Filename-only lookup in pre-built index.
    """
    if not audio_rel:
        return None
    p = Path(audio_rel)
    if p.is_absolute():
        return p if p.exists() else None
    candidate = audio_root / audio_rel
    if candidate.exists():
        return candidate
    name = p.name
    found = index.get(name)
    return Path(found) if found else None
