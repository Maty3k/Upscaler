"""Tests for the corrupt-GPU-output detector that drives the CPU fallback.

These are synthetic and CPU-only (no model download): they verify the heuristic
fires on garbage (NaN, gross colour shift) and stays quiet on correct output, so
the fallback only kicks in when a GPU genuinely fails.
"""

from __future__ import annotations

import torch

from upscaler.engine import output_looks_corrupt


def test_clean_upscale_not_flagged():
    x = torch.rand(1, 3, 16, 16)
    out = torch.nn.functional.interpolate(x, scale_factor=2)  # a faithful 2x
    assert output_looks_corrupt(out, x) is False


def test_mild_enhancement_not_flagged():
    # a small brightness/contrast change must NOT be treated as corruption
    x = torch.rand(1, 3, 16, 16) * 0.6 + 0.2
    out = (x * 1.08).clamp(0, 1)
    out = torch.nn.functional.interpolate(out, scale_factor=2)
    assert output_looks_corrupt(out, x) is False


def test_red_banding_flagged():
    x = torch.full((1, 3, 16, 16), 0.5)
    out = torch.zeros(1, 3, 32, 32)
    out[:, 0] = 1.0  # all-red garbage (the MPS failure signature)
    assert output_looks_corrupt(out, x) is True


def test_nan_flagged():
    x = torch.rand(1, 3, 16, 16)
    out = torch.rand(1, 3, 32, 32)
    out[0, 0, 0, 0] = float("nan")
    assert output_looks_corrupt(out, x) is True


def test_inf_flagged():
    x = torch.rand(1, 3, 16, 16)
    out = torch.rand(1, 3, 32, 32)
    out[0, 1, 5, 5] = float("inf")
    assert output_looks_corrupt(out, x) is True
