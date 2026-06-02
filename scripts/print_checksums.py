"""Download each registered model once and print its SHA-256.

Use the output to pin ``sha256=`` on each ModelSpec in registry.py before a
release, so users get supply-chain integrity on every download.

    python scripts/print_checksums.py
"""

from upscaler.models.registry import MODELS
from upscaler.models.weights import _sha256, ensure_weights

for name, spec in sorted(MODELS.items()):
    path = ensure_weights(spec)
    print(f"{name}: {_sha256(path)}")
