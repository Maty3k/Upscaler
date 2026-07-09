"""Optional GFPGAN face restoration, used as a post-upscale stage.

For each face it detects (OpenCV YuNet), it aligns the face to the standard FFHQ
512 template, restores it with GFPGAN (loaded via ``spandrel`` — no ``basicsr``
dependency), then pastes the result back over the image with a feathered mask.

Best-effort: an image with no detectable faces is returned unchanged. The heavy
dependencies (``spandrel``, ``spandrel_extra_arches``, ``opencv-python``) live
behind the ``[face]`` extra and are imported lazily with a clear message.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from upscaler.engine import load_spandrel, resolve_device
from upscaler.models.registry import DEFAULT_FACE_MODEL, FACE_DETECTOR, FACE_MODELS
from upscaler.models.weights import ensure_weights

# FFHQ 5-point template for a 512 crop, in the order YuNet emits landmarks
# (right eye, left eye, nose, right mouth corner, left mouth corner — which in
# image space already matches left→right, so no reordering is needed).
_TEMPLATE = np.array(
    [[192.98138, 239.94708], [318.90277, 240.19360], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32,
)
_SIZE = 512


def _deps():
    try:
        import cv2  # noqa: F401
        import spandrel  # noqa: F401
        import spandrel_extra_arches  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Face restoration needs extra packages. Install them with: "
            'pip install -e ".[face]"'
        ) from e
    import cv2
    import spandrel
    import spandrel_extra_arches
    return cv2, spandrel, spandrel_extra_arches


class FaceRestorer:
    """Detect + restore faces with GFPGAN. Lazy: the ~349MB GFPGAN net only
    downloads/loads the first time a face is actually found."""

    def __init__(self, model: str = DEFAULT_FACE_MODEL, device: str = "auto"):
        self._cv2, _, _ = _deps()  # validate cv2 + spandrel are installed up front
        if model not in FACE_MODELS:
            raise ValueError(f"Unknown face model {model!r}. Available: {', '.join(FACE_MODELS)}")
        self._spec = FACE_MODELS[model]
        self._device = resolve_device(device)
        self._detector_path = str(ensure_weights(FACE_DETECTOR))
        self._net = None

    def _gfpgan(self):
        if self._net is None:
            # Funnel through the one shared spandrel loader. Face models keep their
            # own forward (detect/align/paste-back), so skip the channel guard.
            lm = load_spandrel(
                ensure_weights(self._spec), self._device, require_channels=None
            )
            self._net = lm.net
        return self._net

    def restore(
        self, image: Image.Image, strength: float = 1.0, fidelity: float = 0.5
    ) -> Image.Image:
        """Return ``image`` with detected faces restored. Unchanged if none.

        ``strength`` blends the restored face back toward the original (likeness).
        ``fidelity`` only applies to models that expose a per-call weight
        (CodeFormer): higher = truer to the original, lower = stronger restoration.
        """
        import torch

        cv2 = self._cv2
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        det = cv2.FaceDetectorYN.create(self._detector_path, "", (w, h), 0.6)
        det.setInputSize((w, h))
        _, faces = det.detect(bgr)
        if faces is None or len(faces) == 0:
            return image

        net = self._gfpgan()
        strength = max(0.0, min(1.0, float(strength)))
        wgt = max(0.0, min(1.0, float(fidelity)))
        uses_fidelity = bool(getattr(self._spec, "fidelity", False))
        out = bgr.astype(np.float32)
        for f in faces:
            lm = f[4:14].reshape(5, 2).astype(np.float32)
            M, _ = cv2.estimateAffinePartial2D(lm, _TEMPLATE, method=cv2.LMEDS)
            if M is None:
                continue
            aligned = cv2.warpAffine(bgr, M, (_SIZE, _SIZE), flags=cv2.INTER_LINEAR)
            t = torch.from_numpy(
                cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(self._device)
            with torch.inference_mode():
                if uses_fidelity:
                    # Drive CodeFormer's native fidelity weight, then match
                    # spandrel's own clamp(0,1) (its __call__ does this for us).
                    out_t = net.model(t, weight=wgt)[0]
                else:
                    out_t = net(t)
                r = out_t.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
            restored = cv2.cvtColor(
                (r * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR
            ).astype(np.float32)
            if strength < 0.999:  # blend back toward the original face
                restored = restored * strength + aligned.astype(np.float32) * (1 - strength)
            inv = cv2.invertAffineTransform(M)
            back = cv2.warpAffine(restored, inv, (w, h), flags=cv2.INTER_LINEAR)
            quad = cv2.warpAffine(np.full((_SIZE, _SIZE), 255, np.uint8), inv, (w, h))
            mask = cv2.erode(quad, np.ones((13, 13), np.uint8))
            mask = cv2.GaussianBlur(mask, (0, 0), 9).astype(np.float32) / 255.0
            # The blur spreads farther than the erosion pulled in, so without
            # this clamp the mask bleeds past the crop, where `back` is black —
            # compositing a dark halo box around every restored face.
            mask = (mask * (quad.astype(np.float32) / 255.0))[..., None]
            out = back * mask + out * (1 - mask)
        return Image.fromarray(
            cv2.cvtColor(out.round().astype(np.uint8), cv2.COLOR_BGR2RGB), "RGB"
        )
