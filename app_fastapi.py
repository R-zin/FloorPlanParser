"""
FastAPI service wrapping the original CubiCasa5k pretrained model.

Loads the model ONCE at startup (on CUDA, explicitly -- fails loudly at
startup if CUDA isn't available rather than silently falling back to CPU,
since that's what was asked for) rather than per-request, which would be
far too slow given model loading alone took a meaningful chunk of the
~40s/image you saw earlier.

Routes:
    POST /parsetojson  -- upload a floorplan image, get back the
                          {"rooms","walls","openings","icons"} JSON.
    POST /visualize     -- upload an image (re-runs inference + draws
                          overlay), OR upload a previously-computed JSON
                          result to draw standalone, OR both (image as
                          background, given JSON drawn on top without
                          re-running inference). Returns a PNG.
    GET  /health        -- liveness/readiness check.

Run this from inside a CubiCasa5k repo clone (same requirement as
run_inference_to_json.py, since it imports from there):
    pip install fastapi uvicorn python-multipart
    uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
"""

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw, ImageFont

from run_inference_to_json import (
    ICON_CLASSES,
    ROOM_CLASSES,
    run_inference,
    load_model,
    polygons_to_json,
    preprocess_image,
)
from floortrans.post_prosessing import get_polygons, split_prediction

MODEL_STATE: dict = {}

# One consistent color per category, used across both room-class colors and
# the wall/opening/icon overlay so the legend stays readable.
ROOM_COLOR = (255, 200, 120, 110)     # translucent orange fill
WALL_COLOR = (40, 40, 40, 255)        # near-black outline
OPENING_COLORS = {"Window": (30, 120, 255, 255), "Door": (220, 40, 40, 255)}
ICON_COLOR = (30, 160, 90, 255)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment, but device is explicitly "
            "set to cuda. Run this on a machine with a GPU, or change DEVICE below "
            "if you actually want CPU fallback."
        )
    device = torch.device("cuda")
    weights_path = MODEL_STATE.get("weights_path", "model_best_val_loss_var.pkl")
    print(f"Loading model from {weights_path} on {device}...")
    model = load_model(weights_path, device)
    MODEL_STATE["model"] = model
    MODEL_STATE["device"] = device
    print("Model loaded. Ready.")
    yield
    MODEL_STATE.clear()


app = FastAPI(title="CubiCasa5k Floorplan Parser", lifespan=lifespan)


def _run_pipeline(image_path: str, max_dim: int = 512, min_icon_confidence: float = 0.3) -> dict:
    model = MODEL_STATE["model"]
    device = MODEL_STATE["device"]

    img_tensor, height, width = preprocess_image(image_path, max_dim=max_dim)
    # run_inference() (from run_inference_to_json.py) handles the forward
    # pass AND moves the prediction to CPU before split_prediction --
    # get_polygons() calls .numpy() internally and errors on a CUDA tensor
    # otherwise. Calling the shared function here instead of duplicating
    # the forward pass means that fix can't drift out of sync between the
    # CLI script and this service.
    heatmaps, rooms, icons = run_inference(model, img_tensor, height, width, device)

    polygons, types, room_polygons, room_types = get_polygons(
        (heatmaps, rooms, icons), threshold=0.2, all_opening_types=[1, 2]
    )
    return polygons_to_json(polygons, types, room_polygons, room_types, min_icon_confidence=min_icon_confidence)


@app.get("/health")
def health():
    return {"status": "ok", "cuda": torch.cuda.is_available(), "model_loaded": "model" in MODEL_STATE}


@app.post("/parsetojson")
async def parse_to_json(
    image: UploadFile = File(...),
    max_dim: int = Form(512),
    min_icon_confidence: float = Form(0.3),
):
    if "model" not in MODEL_STATE:
        raise HTTPException(503, "Model not loaded yet")

    suffix = Path(image.filename).suffix or ".jpg"
    with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await image.read())
        tmp.flush()
        try:
            result = _run_pipeline(tmp.name, max_dim=max_dim, min_icon_confidence=min_icon_confidence)
        except Exception as e:
            raise HTTPException(500, f"Inference failed: {e}") from e

    return JSONResponse(result)


def _iter_contours(polygon_field) -> list[list[list[float]]]:
    """
    Handles BOTH shapes seen in real output: rooms group multiple contours
    per class ([[pt,pt,...], [pt,pt,...]]), while walls/openings/icons are a
    single flat contour ([pt,pt,...]). Normalizes either into a list of
    contours so drawing code doesn't need to care which one it got.
    """
    if not polygon_field:
        return []
    first = polygon_field[0]
    # A "point" is a 2-element list/tuple of numbers. If polygon_field[0] is
    # itself a point, this is a single flat contour -- wrap it.
    if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[0], (int, float)):
        return [polygon_field]
    return polygon_field


def _draw_result(image: Image.Image | None, result: dict) -> Image.Image:
    if image is None:
        # No source image -- figure out a canvas size from the max
        # coordinate seen across all polygons, plus margin.
        max_x, max_y = 100, 100
        for section in ("rooms", "walls", "openings", "icons"):
            for item in result.get(section, []):
                for contour in _iter_contours(item.get("polygon", [])):
                    for pt in contour:
                        max_x = max(max_x, pt[0])
                        max_y = max(max_y, pt[1])
        image = Image.new("RGB", (int(max_x) + 50, int(max_y) + 50), color="white")

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for room in result.get("rooms", []):
        for contour in _iter_contours(room.get("polygon", [])):
            if len(contour) < 3:
                continue
            pts = [(p[0], p[1]) for p in contour]
            draw.polygon(pts, fill=ROOM_COLOR, outline=(0, 0, 0, 255))
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            draw.text((cx, cy), room.get("type", "?"), fill=(0, 0, 0, 255), font=font)

    for wall in result.get("walls", []):
        for contour in _iter_contours(wall.get("polygon", [])):
            if len(contour) < 2:
                continue
            pts = [(p[0], p[1]) for p in contour]
            draw.polygon(pts, outline=WALL_COLOR, fill=(60, 60, 60, 160))

    for opening in result.get("openings", []):
        color = OPENING_COLORS.get(opening.get("type"), (150, 150, 150, 255))
        for contour in _iter_contours(opening.get("polygon", [])):
            if len(contour) < 2:
                continue
            pts = [(p[0], p[1]) for p in contour]
            draw.polygon(pts, outline=color, fill=(*color[:3], 90))
        if "centroid" in opening:
            cx, cy = opening["centroid"]
            r = 4
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    for icon in result.get("icons", []):
        for contour in _iter_contours(icon.get("polygon", [])):
            if len(contour) < 2:
                continue
            pts = [(p[0], p[1]) for p in contour]
            draw.polygon(pts, outline=ICON_COLOR, fill=(*ICON_COLOR[:3], 90))
        if "centroid" in icon:
            cx, cy = icon["centroid"]
            draw.text((cx, cy), icon.get("type", "?"), fill=ICON_COLOR, font=font)

    return Image.alpha_composite(base, overlay).convert("RGB")


@app.post("/visualize")
async def visualize(
    image: UploadFile | None = File(None),
    json_data: str | None = Form(None),
    max_dim: int = Form(512),
    min_icon_confidence: float = Form(0.3),
):
    """
    - image only         -> runs inference on the image, overlays result on it
    - json_data only      -> draws the given result on a blank canvas
    - image + json_data   -> overlays the given (already-computed) result on
                             the image WITHOUT re-running inference
    - neither             -> 400
    """
    if image is None and json_data is None:
        raise HTTPException(400, "Provide at least one of: image, json_data")

    pil_image = None
    if image is not None:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if json_data is not None:
        try:
            result = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"json_data isn't valid JSON: {e}") from e
    else:
        if "model" not in MODEL_STATE:
            raise HTTPException(503, "Model not loaded yet")
        suffix = Path(image.filename).suffix or ".jpg"
        with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            try:
                result = _run_pipeline(tmp.name, max_dim=max_dim, min_icon_confidence=min_icon_confidence)
            except Exception as e:
                raise HTTPException(500, f"Inference failed: {e}") from e

    rendered = _draw_result(pil_image, result)
    buf = io.BytesIO()
    rendered.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")