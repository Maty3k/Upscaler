"""Export pretrained models to ONNX (one-time; requires torch).

The resulting ``.onnx`` files run with onnxruntime alone — no torch needed at
inference time. Exports use dynamic spatial axes so a single file handles any
image size. Files are cached next to the weights and reused.
"""

from __future__ import annotations

from pathlib import Path

from upscaler.models.registry import DeblurSpec, ModelSpec
from upscaler.models.weights import WEIGHTS_DIR, ensure_weights

_DYNAMIC_AXES = {
    "input": {0: "b", 2: "h", 3: "w"},
    "output": {0: "b", 2: "H", 3: "W"},
}


def onnx_path(filename: str) -> Path:
    return WEIGHTS_DIR / (Path(filename).stem + ".onnx")


def _export(module, dummy, dest: Path) -> Path:
    import contextlib
    import io
    import os
    import tempfile

    import torch

    module.eval()
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Export to a temp file, then rename into place: an interrupted/failed
    # export must never leave a truncated .onnx that `dest.exists()` would
    # treat as a valid cache forever after.
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".onnx.part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        # Mute the exporter's console chatter: torch prints status lines with
        # emoji (✅), and on Windows a redirected stdout defaults to cp1252 —
        # the *print* would raise UnicodeEncodeError and kill the export.
        # Let the exporter choose a supported opset — forcing a low one triggers
        # a fragile down-conversion of the Resize/upsample ops.
        with contextlib.redirect_stdout(io.StringIO()):
            torch.onnx.export(
                module,
                dummy,
                str(tmp),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=_DYNAMIC_AXES,
                do_constant_folding=True,
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def export_upscale(spec: ModelSpec) -> Path:
    """Export an RRDBNet upscaler to ONNX (cached). Returns the .onnx path."""
    dest = onnx_path(spec.filename)
    if dest.exists():
        return dest
    import torch

    from upscaler.engine import _load_state_dict
    from upscaler.models.rrdbnet import RRDBNet

    net = RRDBNet(
        scale=spec.scale, num_feat=spec.num_feat,
        num_block=spec.num_block, num_grow_ch=spec.num_grow_ch,
    ).eval()
    net.load_state_dict(_load_state_dict(ensure_weights(spec)), strict=True)
    return _export(net, torch.rand(1, 3, 64, 64), dest)


def export_deblur(spec: DeblurSpec) -> Path:
    """Export a NAFNet deblur model's (divisible-input) body to ONNX (cached)."""
    dest = onnx_path(spec.filename)
    if dest.exists():
        return dest
    import torch

    from upscaler.engine import _load_state_dict
    from upscaler.models.nafnet import NAFNet

    net = NAFNet(
        width=spec.width, middle_blk_num=spec.middle_blk_num,
        enc_blk_nums=spec.enc_blk_nums, dec_blk_nums=spec.dec_blk_nums,
    ).eval()
    net.load_state_dict(_load_state_dict(ensure_weights(spec)), strict=True)

    class _Body(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.body(x)

    # 64 is a multiple of the padder size (16); the numpy wrapper guarantees
    # divisible input at runtime, so the body's shape-preserving graph applies.
    return _export(_Body(net), torch.rand(1, 3, 64, 64), dest)
