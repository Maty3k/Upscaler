"""Recipes — one saved chain of edits, applied to a whole folder.

Every tool here does one thing to one photo. A recipe is the missing piece:
an ordered list of those steps, saved as JSON, so "level it, warm it, sharpen
it, sign it, and squeeze it under 500 KB" becomes one command over two hundred
holiday photos.

A step names a tool and the settings that tool already understands, so nothing
is reimplemented — the recipe just calls the same functions the tabs do, in
order. Unknown keys are ignored rather than fatal, which means a recipe saved
by an older or newer version still runs; a step naming a tool that doesn't
exist is the one thing worth refusing, because silently skipping it would give
you a file that looks right and isn't.

Steps change pixels. What comes out the other end — the format, a size budget,
whether the metadata goes — belongs to the recipe rather than to any step, so
it is set once at the end.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from PIL import Image

from upscaler import adjust, blur, effects, frame, metadata, optimize, sharpen, watermark

SCHEMA = 1

# tool name → (params dataclass, apply function, takes a region mask?)
_TOOLS: "dict[str, tuple]" = {
    "adjust": (adjust.AdjustParams, adjust.apply, True),
    "effects": (effects.EffectParams, effects.apply, True),
    "blur": (blur.BlurParams, blur.apply, True),
    "sharpen": (sharpen.SharpenParams, sharpen.apply, True),
    "crop": (frame.FrameParams, frame.apply, False),
    "watermark": (watermark.WatermarkParams, watermark.apply, False),
}
TOOLS = list(_TOOLS)


@dataclass
class Step:
    tool: str
    params: dict = field(default_factory=dict)
    region: dict = field(default_factory=dict)   # a blur.MaskParams, for tools that take one

    def describe(self) -> str:
        bits = [f"{k} {v}" for k, v in sorted(self.params.items())][:3]
        where = self.region.get("shape")
        text = f"{self.tool}" + (f" ({', '.join(bits)})" if bits else "")
        return text + (f" · {where}" if where and where != "whole" else "")


@dataclass
class Recipe:
    name: str = "My recipe"
    steps: "list[Step]" = field(default_factory=list)
    # What the finished file should be. Empty means "leave it as it was".
    output_format: str = ""      # e.g. "JPEG", "WebP"
    target_size: str = ""        # e.g. "500KB" — a size budget for the output
    strip_metadata: bool = False

    def describe(self) -> str:
        if not self.steps and not self.output_format and not self.target_size:
            return f"{self.name}: does nothing"
        bits = [s.describe() for s in self.steps]
        tail = []
        if self.output_format:
            tail.append(f"save as {self.output_format}")
        if self.target_size:
            tail.append(f"under {self.target_size}")
        if self.strip_metadata:
            tail.append("metadata removed")
        return f"{self.name}: " + " → ".join(bits + tail)


class RecipeError(ValueError):
    """A recipe that can't be run as written."""


# ── saving and loading ────────────────────────────────────────────────────────
def to_dict(recipe: Recipe) -> dict:
    return {"schema": SCHEMA, "name": recipe.name,
            "output_format": recipe.output_format, "target_size": recipe.target_size,
            "strip_metadata": recipe.strip_metadata,
            "steps": [asdict(s) for s in recipe.steps]}


def to_json(recipe: Recipe) -> str:
    return json.dumps(to_dict(recipe), indent=2)


def from_dict(data: dict) -> Recipe:
    """Build a recipe from parsed JSON, tolerating unknown keys."""
    if not isinstance(data, dict):
        raise RecipeError("a recipe must be a JSON object")
    steps = []
    for raw in data.get("steps") or []:
        if not isinstance(raw, dict):
            raise RecipeError(f"each step must be an object, got {type(raw).__name__}")
        tool = str(raw.get("tool", "")).strip()
        if tool not in _TOOLS:
            raise RecipeError(
                f"step names an unknown tool {tool!r} — expected one of {', '.join(TOOLS)}")
        steps.append(Step(tool=tool, params=dict(raw.get("params") or {}),
                          region=dict(raw.get("region") or {})))
    return Recipe(
        name=str(data.get("name") or "Recipe"),
        steps=steps,
        output_format=str(data.get("output_format") or ""),
        target_size=str(data.get("target_size") or ""),
        strip_metadata=bool(data.get("strip_metadata")),
    )


def from_json(text: str) -> Recipe:
    try:
        return from_dict(json.loads(text))
    except json.JSONDecodeError as e:
        raise RecipeError(f"that isn't valid JSON: {e}") from e


def load(path: "str | Path") -> Recipe:
    return from_json(Path(path).read_text())


def save(recipe: Recipe, path: "str | Path") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_json(recipe))
    return p


# ── running ───────────────────────────────────────────────────────────────────
def _build(cls, values: dict):
    """A params object from a dict, ignoring keys the class doesn't have — so a
    recipe written against another version still runs."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in known})


def _region_for(step: Step, img: Image.Image, depth_model: "str | None" = None):
    """The mask a step asks for, resolving the ones that need the image itself:
    faces have to be detected and depth has to be estimated."""
    if not step.region:
        return None
    mask = _build(blur.MaskParams, step.region)
    if mask.shape == "faces" and not mask.faces:
        from upscaler import face

        mask = replace(mask, faces=[f.box() for f in face.detect_faces(img)])
    elif mask.shape == blur.DEPTH and mask.depth is None:
        from upscaler import depth as depth_tools

        kw = {"model": depth_model} if depth_model else {}
        mask = replace(mask, depth=depth_tools.estimate(img, **kw))
    elif mask.shape == "painted" and mask.painted is None:
        raise RecipeError("a painted region can't be saved in a recipe — use a "
                          "shape, faces or depth instead")
    return mask


def run_steps(img: Image.Image, recipe: Recipe, depth_model: "str | None" = None,
              progress=None) -> Image.Image:
    """Apply every step in order and return the finished picture."""
    out = img
    total = max(1, len(recipe.steps))
    for i, step in enumerate(recipe.steps):
        cls, fn, takes_region = _TOOLS[step.tool]
        params = _build(cls, step.params)
        if progress:
            progress((i + 1) / total, desc=f"{step.tool} ({i + 1}/{total})")
        if takes_region:
            region = _region_for(step, out, depth_model)
            out = fn(out, params, region) if region is not None else fn(out, params)
        else:
            out = fn(out, params)
    return out


@dataclass
class RecipeResult:
    image: Image.Image
    data: "bytes | None" = None     # set when the output settings produced a file
    fmt: str = ""
    extension: str = "png"
    note: str = ""


def run(img: Image.Image, recipe: Recipe, depth_model: "str | None" = None,
        progress=None) -> RecipeResult:
    """Run the steps, then honour the recipe's output settings.

    The finished bytes come back rather than a path, so the caller decides
    where they land — a folder, a ZIP, or the Library.
    """
    out = run_steps(img, recipe, depth_model, progress)
    result = RecipeResult(image=out)

    if recipe.target_size:
        budget = optimize.OptimizeParams(
            target=recipe.target_size, fmt=recipe.output_format or optimize.AUTO)
        got = optimize.optimize(out, budget)
        result.data, result.fmt, result.extension = got.data, got.fmt, got.extension
        result.note = optimize.describe(got)
    elif recipe.output_format:
        from upscaler.convert import convert, extension_for

        result.data = convert(out, recipe.output_format, quality=92)
        result.fmt = recipe.output_format
        result.extension = extension_for(recipe.output_format)

    if recipe.strip_metadata and result.data:
        # Only the encoders above can leave anything behind, and this is
        # lossless for JPEG and PNG, so it costs nothing to be thorough.
        stripped = metadata.strip(result.data, metadata.REMOVE_ALL)
        result.data = stripped.data
    return result


# ── ready-made recipes ────────────────────────────────────────────────────────
BUILT_IN: "dict[str, Recipe]" = {
    "Web-ready": Recipe(
        name="Web-ready",
        steps=[Step("adjust", {"exposure": 3, "contrast": 8, "vibrance": 15,
                               "clarity": 8}),
               Step("sharpen", {"kind": "smart", "amount": 70, "radius": 0.8})],
        output_format="WebP", target_size="400KB", strip_metadata=True,
    ),
    "Print-safe portrait": Recipe(
        name="Print-safe portrait",
        steps=[Step("adjust", {"exposure": 4, "shadows": 15, "vibrance": 12}),
               Step("sharpen", {"kind": "smart", "amount": 90, "radius": 1.0,
                                "protect_highlights": 25})],
        strip_metadata=True,
    ),
    "Instagram square": Recipe(
        name="Instagram square",
        steps=[Step("crop", {"aspect": "Square · 1:1"}),
               Step("adjust", {"contrast": 10, "vibrance": 18, "clarity": 12}),
               Step("sharpen", {"amount": 70})],
        output_format="JPEG", target_size="1MB", strip_metadata=True,
    ),
    "Film look": Recipe(
        name="Film look",
        steps=[Step("adjust", {"contrast": -6, "black_point": 6, "temperature": 12,
                               "saturation": -8}),
               Step("effects", {"grain": 24, "grain_size": 1.6, "halation": 30,
                                "vignette": 22})],
        strip_metadata=True,
    ),
    "Blur every face": Recipe(
        name="Blur every face",
        steps=[Step("blur", {"kind": "pixelate", "strength": 50},
                    {"shape": "faces", "feather": 4, "face_pad": 30})],
        strip_metadata=True,
    ),
    "Portrait depth of field": Recipe(
        name="Portrait depth of field",
        steps=[Step("blur", {"kind": "lens", "strength": 45, "highlights": 40},
                    {"shape": "depth", "focus": 70, "dof": 25, "feather": 2})],
    ),
    "Sign and shrink": Recipe(
        name="Sign and shrink",
        steps=[Step("watermark", {"text": "© Your Name", "size": 3.5,
                                  "position": "bottom right", "opacity": 75})],
        output_format="JPEG", target_size="500KB", strip_metadata=True,
    ),
}
BUILT_IN_NAMES = list(BUILT_IN)


def built_in(name: str) -> Recipe:
    """A copy of a ready-made recipe, so editing it can't affect the original."""
    if name not in BUILT_IN:
        raise RecipeError(f"unknown recipe {name!r} — "
                          f"expected one of {', '.join(BUILT_IN_NAMES)}")
    return from_dict(to_dict(BUILT_IN[name]))
