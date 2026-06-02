"""Weight integrity tests — offline, no real downloads."""

import pytest

import upscaler.models.weights as weights
from upscaler.models.registry import MODELS, DeblurSpec


def test_all_registered_weights_are_pinned():
    from upscaler.models.registry import DEBLUR_MODELS
    for name, spec in {**MODELS, **DEBLUR_MODELS}.items():
        assert spec.sha256 and len(spec.sha256) == 64, f"{name} missing/short sha256"


def test_checksum_mismatch_rejects_and_removes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(weights, "WEIGHTS_DIR", tmp_path)
    spec = DeblurSpec(
        name="fake", url="http://example.invalid/x.pth", filename="x.pth",
        width=16, middle_blk_num=1, enc_blk_nums=(1,), dec_blk_nums=(1,),
        sha256="0" * 64,  # cannot match real content
    )
    # Pre-place a "downloaded" file so ensure_weights skips the network and
    # goes straight to verification.
    (tmp_path / "x.pth").write_bytes(b"not the real weights")

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        weights.ensure_weights(spec)
    assert not (tmp_path / "x.pth").exists()  # corrupt file removed
