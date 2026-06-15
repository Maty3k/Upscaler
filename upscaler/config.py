"""User preferences, persisted to ``~/.upscaler/config.json``.

Loaded once at startup to seed the GUI's default device / model / save-folder,
and written from the Settings tab. Deliberately tiny and best-effort: unknown
keys are ignored and any read/write error falls back to the built-in defaults,
so a corrupt or missing file can never stop the app from starting.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_PATH = Path(
    os.environ.get("UPSCALER_CONFIG", str(Path.home() / ".upscaler" / "config.json"))
)

DEFAULTS: "dict[str, str]" = {
    "device": "auto",
    "model": "realesrgan-x4plus",
    "output_dir": "",
}


def load() -> "dict[str, str]":
    """Return the saved preferences merged over the defaults."""
    cfg = dict(DEFAULTS)
    try:
        if CONFIG_PATH.is_file():
            data = json.loads(CONFIG_PATH.read_text())
            if isinstance(data, dict):
                cfg.update({k: data[k] for k in DEFAULTS if k in data})
    except (OSError, ValueError):
        pass
    return cfg


def save(**values: str) -> bool:
    """Update the known preference keys and write them back. Returns success."""
    cfg = load()
    cfg.update({k: v for k, v in values.items() if k in DEFAULTS})
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        return True
    except OSError:
        return False
