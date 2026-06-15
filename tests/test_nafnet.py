"""NAFNet deblur architecture/registry tests — CPU, no weight download."""

import pytest
import torch

from upscaler.models.nafnet import NAFNet, SimpleGate
from upscaler.models.registry import (
    DEBLUR_MODELS,
    DEFAULT_DEBLUR_MODEL,
    resolve_deblur_model,
)


def _tiny_net():
    return NAFNet(
        img_channel=3, width=16, middle_blk_num=1,
        enc_blk_nums=(1, 1), dec_blk_nums=(1, 1),
    ).eval()


def test_nafnet_preserves_resolution():
    net = _tiny_net()
    x = torch.rand(1, 3, 32, 32)
    with torch.inference_mode():
        y = net(x)
    assert y.shape == x.shape  # deblur is scale-1


@pytest.mark.parametrize("h,w", [(30, 30), (45, 53), (17, 64)])
def test_nafnet_handles_non_multiple_sizes(h, w):
    """Internal padding must accept any size and crop back to the original."""
    net = _tiny_net()
    with torch.inference_mode():
        y = net(torch.rand(1, 3, h, w))
    assert y.shape == (1, 3, h, w)


def test_simple_gate_halves_channels():
    out = SimpleGate()(torch.rand(1, 8, 4, 4))
    assert out.shape == (1, 4, 4, 4)


def test_resolve_deblur_model():
    assert resolve_deblur_model().name == DEFAULT_DEBLUR_MODEL
    assert resolve_deblur_model("nafnet-gopro-width64").width == 64
    assert resolve_deblur_model("nafnet-gopro-width32").width == 32
    with pytest.raises(ValueError):
        resolve_deblur_model("nope")


def test_deblur_specs_use_confirmed_config():
    # Confirmed block layouts from the official NAFNet option files: the two
    # GoPro deblur checkpoints share one config; SIDD denoise uses a deeper one.
    # (width, middle_blk_num, enc_blk_nums, dec_blk_nums)
    confirmed = {
        "nafnet-gopro-width64": (64, 1, (1, 1, 1, 28), (1, 1, 1, 1)),
        "nafnet-gopro-width32": (32, 1, (1, 1, 1, 28), (1, 1, 1, 1)),
        "nafnet-sidd-width64": (64, 12, (2, 2, 4, 8), (2, 2, 2, 2)),
    }
    # A new deblur model must come with its confirmed config here.
    assert set(DEBLUR_MODELS) == set(confirmed)
    for name, (width, middle, enc, dec) in confirmed.items():
        spec = DEBLUR_MODELS[name]
        assert spec.width == width, name
        assert spec.middle_blk_num == middle, name
        assert spec.enc_blk_nums == enc, name
        assert spec.dec_blk_nums == dec, name
