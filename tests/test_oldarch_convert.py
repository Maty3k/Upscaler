"""Tests for the old-arch ESRGAN -> Real-ESRGAN RRDBNet key converter that lets
community models (4x-UltraSharp, Remacri, NMKD, …) load into the vendored net."""

import re

from upscaler.engine import _convert_esrgan_oldarch
from upscaler.models.rrdbnet import RRDBNet


def _new_to_old(sd, nb):
    """Reverse the converter: rename new-arch keys to the old `model.*` layout."""
    out = {}
    for k, v in sd.items():
        m = re.match(r"^body\.(\d+)\.rdb(\d+)\.conv(\d+)\.(weight|bias)$", k)
        if m:
            i, j, c, wb = m.groups()
            out[f"model.1.sub.{i}.RDB{j}.conv{c}.0.{wb}"] = v
        elif k.startswith("conv_first."):
            out["model.0." + k.split(".", 1)[1]] = v
        elif k.startswith("conv_body."):
            out[f"model.1.sub.{nb}." + k.split(".", 1)[1]] = v
        elif k.startswith("conv_up1."):
            out["model.3." + k.split(".", 1)[1]] = v
        elif k.startswith("conv_up2."):
            out["model.6." + k.split(".", 1)[1]] = v
        elif k.startswith("conv_hr."):
            out["model.8." + k.split(".", 1)[1]] = v
        elif k.startswith("conv_last."):
            out["model.10." + k.split(".", 1)[1]] = v
        else:
            out[k] = v
    return out


def test_oldarch_converts_and_strict_loads():
    nb = 2
    net = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=8, num_block=nb, num_grow_ch=4)
    orig = net.state_dict()
    old = _new_to_old(orig, nb)
    assert any(k.startswith("model.") for k in old)  # genuinely old-arch now
    assert not any(k.startswith("conv_first") for k in old)

    converted = _convert_esrgan_oldarch(old)
    assert set(converted) == set(orig)  # exact key set recovered
    # and it loads strictly back into the vendored generator
    net2 = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=8, num_block=nb, num_grow_ch=4)
    net2.load_state_dict(converted, strict=True)


def test_newarch_passthrough_is_untouched():
    net = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=8, num_block=2, num_grow_ch=4)
    sd = net.state_dict()
    assert _convert_esrgan_oldarch(sd) is sd  # no `model.*` keys -> returned as-is


def test_community_models_registered_and_pinned():
    from upscaler.models.registry import MODELS

    for name in ("4x-ultrasharp", "4x-remacri", "4x-nmkd-siax", "4x-nmkd-superscale"):
        assert name in MODELS, f"{name} missing from registry"
        spec = MODELS[name]
        assert spec.scale == 4 and spec.num_block == 23
        assert spec.sha256 and len(spec.sha256) == 64
