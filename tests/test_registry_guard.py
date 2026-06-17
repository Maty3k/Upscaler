"""Supply-chain guard: every shipped weight must be sha256-pinned over https.

This consolidates (and strengthens: 64-hex + https, not just length) the checks
that were previously split across test_weights.py and test_face.py, and covers
every registry via the single ``iter_pinned_specs`` aggregator so new models
added in later batches can't silently escape the guard.
"""

import re

import pytest

from upscaler.models import weights as W
from upscaler.models.registry import ModelSpec, iter_pinned_specs

_HEX64 = re.compile(r"[0-9a-f]{64}")


def test_every_pinned_spec_is_https_and_sha256():
    specs = list(iter_pinned_specs())
    assert specs, "aggregator yielded nothing — registry wiring is broken"
    for spec in specs:
        assert _HEX64.fullmatch(spec.sha256 or ""), f"{spec.name}: sha256 not 64-hex"
        assert (spec.url or "").startswith("https://"), f"{spec.name}: url not https"
        assert spec.filename, f"{spec.name}: missing filename"


def test_checksum_mismatch_removes_modelspec_file(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "WEIGHTS_DIR", tmp_path)
    spec = ModelSpec(
        name="fake", url="https://example.test/x.pth", scale=4,
        filename="x.pth", sha256="0" * 64,
    )
    f = tmp_path / "x.pth"
    f.write_bytes(b"not the real weights")  # present -> no download attempted
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        W.ensure_weights(spec)
    assert not f.exists()  # corrupt file is removed so a retry re-downloads
