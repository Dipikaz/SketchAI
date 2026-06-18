"""
Procreate (.procreate) file exporter for SketchAI.

Strategy: inject our 6 layers into a known-good reference .procreate file
(silicate/Reference_Blend_File.procreate, downloaded from GitHub).
We reuse the reference's proven NSKeyedArchiver structure — colorProfile,
ValkyrieColorProfile, SilicaDocument class desc, composite layer format —
and only append our layer objects + update the document's layer pointers.

This avoids rebuilding the archive from scratch, which was fragile.
"""

from __future__ import annotations

import io
import os
import struct
import urllib.request
import uuid
import zipfile
import plistlib
from pathlib import Path

import lzo
import numpy as np
from PIL import Image


CHUNK_SIZE = 256

LAYER_SPECS = [
    ("layer_6_empty",            "Your Drawing",     1.00, False),
    ("layer_5_linework",         "Linework",         0.70, False),
    ("layer_4_foreground",       "Foreground",       0.50, False),
    ("layer_3_midground",        "Midground",        0.40, False),
    ("layer_2_background",       "Background",       0.30, False),
    ("layer_1_perspective_grid", "Perspective Grid", 0.20, True),
]

_REFERENCE_URL  = "https://github.com/Avarel/silicate/raw/master/demo_files/Reference_Blend_File.procreate"
_REFERENCE_PATH = Path(os.path.expanduser("~/.cache/sketchai/Reference_Blend_File.procreate"))
_IDENTITY_TRANSFORM = struct.pack("<16d", 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_procreate(layers_dir: str | Path, output_path: str | Path) -> Path:
    layers_dir  = Path(layers_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load layer PNGs
    layers: list[dict] = []
    canvas_w = canvas_h = 0
    for stem, name, opacity, locked in LAYER_SPECS:
        png_path = layers_dir / f"{stem}.png"
        if not png_path.exists():
            raise FileNotFoundError(f"Missing: {png_path}")
        arr = np.array(Image.open(png_path).convert("RGBA"), dtype=np.uint8)
        canvas_h, canvas_w = arr.shape[:2]
        layer_uuid = str(uuid.uuid4()).upper()
        layers.append({"uuid": layer_uuid, "name": name,
                        "opacity": opacity, "locked": locked, "array": arr})
        print(f"  {stem}.png  →  {layer_uuid[:8]}…  ({name})")

    ref_path = _ensure_reference()
    archive_bytes = _inject_layers(ref_path, layers, canvas_w, canvas_h)
    thumbnail_bytes = _make_thumbnail(layers, canvas_w, canvas_h)

    # Build composite UUID from the reference (already in the archive)
    with zipfile.ZipFile(ref_path) as zf:
        ref_archive = plistlib.loads(zf.read("Document.archive"))
    ref_objects = ref_archive["$objects"]
    ref_root    = ref_objects[ref_archive["$top"]["root"].data]
    composite_uuid = ref_objects[ref_objects[ref_root["composite"].data]["UUID"].data]

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Document.archive", archive_bytes)
        zf.writestr("QuickLook/Thumbnail.png", thumbnail_bytes)
        for layer in layers:
            _write_chunks(zf, layer["array"], layer["uuid"])
        # Blank composite tiles so Procreate can render them on open
        _write_chunks(zf, np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8), composite_uuid)

    kb = output_path.stat().st_size // 1024
    print(f"\n  → {output_path}  ({kb} KB)")
    return output_path


# ---------------------------------------------------------------------------
# Reference file
# ---------------------------------------------------------------------------

def _ensure_reference() -> Path:
    if _REFERENCE_PATH.exists():
        return _REFERENCE_PATH
    # Check /tmp first (left over from previous session download)
    tmp_ref = Path("/tmp/Reference_Blend_File.procreate")
    if tmp_ref.exists():
        _REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(tmp_ref, _REFERENCE_PATH)
        return _REFERENCE_PATH
    _REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading reference file…")
    import ssl, shutil
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    with opener.open(_REFERENCE_URL) as resp, open(_REFERENCE_PATH, "wb") as f:
        shutil.copyfileobj(resp, f)
    print(f"  Cached at {_REFERENCE_PATH}")
    return _REFERENCE_PATH


# ---------------------------------------------------------------------------
# Archive injection — the core of the safe approach
# ---------------------------------------------------------------------------

def _inject_layers(
    ref_path: Path,
    layers: list[dict],
    canvas_w: int,
    canvas_h: int,
) -> bytes:
    """
    Load the reference archive, append our 6 layer objects, update the
    SilicaDocument pointers, and return a new binary plist.

    We NEVER rebuild the archive from scratch — we reuse:
      - All existing class descriptions (SilicaLayer/ValkyrieLayer, SilicaDocument/
        ValkyrieDocument, ValkyrieColorProfile, NSMutableArray, …)
      - The colorProfile object (ValkyrieColorProfile + ICC data)
      - The composite SilicaLayer object (just update its dimensions)
      - All document-level metadata (DPI, orientation, video flags, etc.)

    We only ADD new objects for our layers and update the root pointers.
    Class UIDs confirmed from reference:
      [18]  SilicaLayer  (classes: SilicaLayer, ValkyrieLayer, NSObject)
      [363] NSMutableArray
      [391] SilicaDocument  (classes: SilicaDocument, ValkyrieDocument, NSObject)
    """
    with zipfile.ZipFile(ref_path) as zf:
        archive = plistlib.loads(zf.read("Document.archive"))

    objects  = archive["$objects"]
    root_uid = archive["$top"]["root"]   # UID(1)
    root     = objects[root_uid.data]    # SilicaDocument dict — we mutate this in place

    # Reuse proven class UIDs from the reference
    uid_cls_layer = plistlib.UID(18)   # SilicaLayer/ValkyrieLayer/NSObject
    uid_cls_array = plistlib.UID(363)  # NSMutableArray

    def add(obj) -> plistlib.UID:
        idx = len(objects)
        objects.append(obj)
        return plistlib.UID(idx)

    # Add identity transform once (shared by all our layers)
    uid_transform = add(_IDENTITY_TRANSFORM)

    # Update canvas size string (reference has "{1836, 2118}")
    uid_size = add(f"{{{canvas_w}, {canvas_h}}}")

    # Update composite layer dimensions in place
    comp = objects[root["composite"].data]
    comp["sizeWidth"]         = canvas_w
    comp["sizeHeight"]        = canvas_h
    comp["contentsRectValid"] = False
    comp["contentsRect"]      = add(struct.pack("<4d", 0.0, 0.0, float(canvas_w), float(canvas_h)))
    comp["transform"]         = uid_transform

    # Build our 6 layers
    layer_uids: list[plistlib.UID] = []
    for layer in layers:
        uid_luuid = add(layer["uuid"])
        uid_lname = add(layer["name"])
        uid_lrect = add(struct.pack("<4d", 0.0, 0.0, float(canvas_w), float(canvas_h)))
        uid_layer = add({
            "$class":              uid_cls_layer,
            "UUID":                uid_luuid,
            "name":                uid_lname,
            "document":            root_uid,          # back-ref to root (UID 1)
            "opacity":             float(layer["opacity"]),
            "locked":              bool(layer["locked"]),
            "hidden":              False,
            "blend":               0,
            "extendedBlend":       0,
            "clipped":             False,
            "type":                0,
            "version":             3,
            "sizeWidth":           canvas_w,
            "sizeHeight":          canvas_h,
            "contentsRect":        uid_lrect,
            "contentsRectValid":   True,
            "transform":           uid_transform,
            "preserve":            False,
            "private":             False,
            "perspectiveAssisted": False,
            "animationHeldLength": 0,
            "mask":                plistlib.UID(0),
            "bundledImagePath":    plistlib.UID(0),
            "bundledMaskPath":     plistlib.UID(0),
            "bundledVideoPath":    plistlib.UID(0),
            "text":                plistlib.UID(0),
            "textPDF":             plistlib.UID(0),
            "textureSet":          plistlib.UID(0),
            "videoTime":           plistlib.UID(0),
        })
        layer_uids.append(uid_layer)

    # New layers NSMutableArray
    uid_layers_arr = add({
        "$class":     uid_cls_array,
        "NS.objects": layer_uids,
    })

    # White background — reference has teal bg + backgroundHidden=True which makes
    # the canvas appear dark when our transparent-alpha layers are loaded.
    uid_bg_rgba = add(struct.pack("<4f", 1.0, 1.0, 1.0, 1.0))   # white RGBA
    uid_bg_hsba = add(struct.pack("<4f", 0.0, 0.0, 1.0, 1.0))   # white HSBA

    # Mutate root SilicaDocument (in place — all other fields preserved)
    root["size"]                  = uid_size
    root["layers"]                = uid_layers_arr
    root["unwrappedLayers"]       = uid_layers_arr
    root["selectedLayer"]         = layer_uids[0]
    root["primaryItem"]           = layer_uids[0]
    root["selectedSamplerLayer"]  = plistlib.UID(0)
    root["mask"]                  = plistlib.UID(0)
    root["solo"]                  = plistlib.UID(0)
    root["animation"]             = plistlib.UID(0)
    root["name"]                  = add("SketchAI Export")
    root["backgroundColor"]       = uid_bg_rgba
    root["backgroundColorHSBA"]   = uid_bg_hsba
    root["backgroundHidden"]      = False   # reference had True → dark canvas

    return plistlib.dumps(archive, fmt=plistlib.FMT_BINARY)


# ---------------------------------------------------------------------------
# Chunk I/O — LZO1X-1 raw stream (no python-lzo header)
# ---------------------------------------------------------------------------

def _write_chunks(zf: zipfile.ZipFile, rgba: np.ndarray, layer_uuid: str) -> None:
    h, w = rgba.shape[:2]
    n_cols = (w + CHUNK_SIZE - 1) // CHUNK_SIZE
    n_rows = (h + CHUNK_SIZE - 1) // CHUNK_SIZE
    for col in range(n_cols):
        for row in range(n_rows):
            x0, y0 = col * CHUNK_SIZE, row * CHUNK_SIZE
            x1, y1 = min(x0 + CHUNK_SIZE, w), min(y0 + CHUNK_SIZE, h)
            tile = np.zeros((CHUNK_SIZE, CHUNK_SIZE, 4), dtype=np.uint8)
            tile[:y1 - y0, :x1 - x0] = rgba[y0:y1, x0:x1]
            # Strip python-lzo's 5-byte header to get raw LZO1X-1 stream
            zf.writestr(f"{layer_uuid}/{col}~{row}.chunk",
                        lzo.compress(tile.tobytes(), 1)[5:])


# ---------------------------------------------------------------------------
# QuickLook thumbnail
# ---------------------------------------------------------------------------

def _make_thumbnail(layers: list[dict], canvas_w: int, canvas_h: int) -> bytes:
    composite = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
    for layer in reversed(layers):
        arr  = layer["array"].astype(np.float32) / 255.0
        alpha = arr[..., 3:4] * layer["opacity"]
        composite[..., :3] = arr[..., :3] * alpha + composite[..., :3] * (1 - alpha)
        composite[..., 3:]  = alpha + composite[..., 3:] * (1 - alpha)
    img = Image.fromarray((np.clip(composite, 0, 1) * 255).astype(np.uint8), "RGBA")
    img.thumbnail((256, 256), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    layers_dir  = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser("~/sketchai/test-outputs/layers")
    output_path = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.expanduser("~/sketchai/test-outputs/output.procreate")

    print(f"Layers : {layers_dir}")
    print(f"Output : {output_path}\n")
    export_procreate(layers_dir, output_path)
