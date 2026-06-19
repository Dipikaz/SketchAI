"""
ComfyUI API client for SketchAI.

Uses a self-contained text-to-image workflow (no ControlNet reference image
required). The lineart look comes entirely from the prompt tokens. ControlNet
can be layered on top in a future iteration once image-upload is wired in.

ComfyUI API:
  POST /prompt              → submit workflow, get prompt_id
  GET  /history/{prompt_id} → poll until completed = true
  GET  /view?filename=...   → download the output PNG
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path

import httpx

COMFYUI_BASE = "http://127.0.0.1:8188"
POLL_INTERVAL = 2        # seconds between history polls
TIMEOUT       = 300      # max seconds to wait for ComfyUI

# ---------------------------------------------------------------------------
# Minimal T2I workflow (API node format, not the UI JSON format)
# Node 7 is SaveImage — history response keys on this node's outputs.
# ---------------------------------------------------------------------------

def _build_workflow(positive: str, negative: str) -> dict:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 768, "height": 768, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model":        ["1", 0],
                "positive":     ["2", 0],
                "negative":     ["3", 0],
                "latent_image": ["4", 0],
                "seed":         random.randint(0, 2**32 - 1),
                "steps":        20,
                "cfg":          7.0,
                "sampler_name": "euler",
                "scheduler":    "normal",
                "denoise":      1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "sketchai"},
        },
    }


async def generate_sketch(
    positive: str,
    negative: str,
    output_dir: Path,
) -> Path:
    """
    Submit a T2I job to ComfyUI and wait for the output PNG.
    Returns the path of the saved PNG inside output_dir.
    Raises RuntimeError if ComfyUI is unreachable, times out, or errors.
    """
    workflow = _build_workflow(positive, negative)

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                f"{COMFYUI_BASE}/prompt",
                json={"prompt": workflow},
            )
            resp.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                "ComfyUI is not running — start it with: "
                "cd ~/ComfyUI && python main.py"
            )

    prompt_id: str = resp.json()["prompt_id"]

    # Poll history until the job completes
    elapsed = 0
    async with httpx.AsyncClient(timeout=10) as client:
        while elapsed < TIMEOUT:
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

            history_resp = await client.get(f"{COMFYUI_BASE}/history/{prompt_id}")
            history = history_resp.json()

            if prompt_id not in history:
                continue  # not done yet

            job = history[prompt_id]
            status = job.get("status", {})

            if status.get("status_str") == "error":
                msgs = [m["details"] for m in status.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(f"ComfyUI error: {msgs or status}")

            if not status.get("completed"):
                continue

            # Find the output image in node 7 (SaveImage)
            outputs = job.get("outputs", {})
            for node_outputs in outputs.values():
                for img in node_outputs.get("images", []):
                    img_resp = await client.get(
                        f"{COMFYUI_BASE}/view",
                        params={
                            "filename": img["filename"],
                            "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output"),
                        },
                    )
                    img_resp.raise_for_status()
                    out_path = output_dir / "sketch.png"
                    out_path.write_bytes(img_resp.content)
                    return out_path

            raise RuntimeError("ComfyUI completed but no output image found")

    raise RuntimeError(f"ComfyUI timed out after {TIMEOUT}s")
