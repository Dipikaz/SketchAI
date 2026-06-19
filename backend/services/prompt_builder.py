from __future__ import annotations

_STYLE_TOKENS: dict[str, str] = {
    "construction lines": "construction lines, architectural sketch, thin precise lines",
    "ink":                "ink sketch, pen and ink, bold strokes",
    "pencil":             "pencil sketch, graphite, hatching",
}

_NEGATIVE_BASE = (
    "color, shading, blur, noise, watermark, photorealistic, "
    "people, ugly, deformed, duplicate, low quality"
)


def build_prompt(scene: str, style: str, vanishing_points: int) -> tuple[str, str]:
    """Return (positive_prompt, negative_prompt) for ComfyUI."""
    style_tokens = _STYLE_TOKENS.get(style, style)
    vp_token = (
        "single vanishing point perspective"
        if vanishing_points == 1
        else f"{vanishing_points} point perspective"
    )
    positive = (
        f"{scene}, lineart sketch, clean black lines on white background, "
        f"no shading, no color, {style_tokens}, {vp_token}"
    )
    return positive, _NEGATIVE_BASE
