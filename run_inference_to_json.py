"""
Runs the ACTUAL CubiCasa5k pretrained model (model_best_val_loss_var.pkl) on a
floorplan image and converts the output into JSON.

IMPORTANT — where this must run:
    This script imports `floortrans`, which only exists inside a clone of the
    original CubiCasa5k repo (https://github.com/CubiCasa/CubiCasa5k). Run it
    from that repo's root directory (or add it to PYTHONPATH), with their
    pinned dependencies (Python 3.6.5, PyTorch 1.0.0) — easiest via their
    Docker setup, since newer PyTorch/Python versions aren't guaranteed
    compatible with a 2018-era codebase.

Setup (from inside a clone of CubiCasa/CubiCasa5k):
    1. Download model_best_val_loss_var.pkl from the Google Drive link in
       their README and place it in the repo root.
    2. Copy this file into the repo root too (so `floortrans` is importable).
    3. Run:
       python run_inference_to_json.py --image path/to/floorplan.jpg --out result.json

What this does NOT use:
    model_1427 / model_1427.pth. That's the base hourglass backbone's generic
    pretrained initialization checkpoint used before CubiCasa-specific
    training — not a floorplan parser itself. See the inline comments below
    for exactly where the real trained weights get loaded instead.

Known uncertainty, flagged explicitly rather than guessed silently:
    I could not confirm the exact return structure of
    `floortrans.post_prosessing.get_polygons()` from source (list of point
    arrays? per-polygon dicts?) — only that it returns
    (polygons, types, room_polygons, room_types). `_debug_inspect_polygons()`
    below prints the real shapes on your first run so you can adjust
    `polygons_to_json()` if the assumed structure doesn't match. Room class
    names are read directly from the real `rooms_selected`/`icons_selected`
    ordering (confirmed against source), so ROOM_CLASSES/ICON_CLASSES here are
    accurate regardless of the polygon-structure uncertainty.
"""

import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

# These only resolve inside a CubiCasa5k repo clone.
from floortrans.models import get_model
from floortrans.post_prosessing import get_polygons, split_prediction

# Confirmed exact class lists/order — matches rooms_selected/icons_selected
# in floortrans/loaders/house.py, and matches split=[21,12,11] (44 total).
ROOM_CLASSES = [
    "Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room",
    "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined",
]
ICON_CLASSES = [
    "No Icon", "Window", "Door", "Closet", "Electrical Appliance", "Toilet",
    "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney",
]
N_CLASSES = 44
SPLIT = [21, 12, 11]  # heatmaps, rooms, icons


def load_model(weights_path: str, device: torch.device):
    """
    Loads the REAL trained CubiCasa5k model. This is the exact sequence their
    own eval.py / samples.ipynb / community forks use:
      1. Instantiate the hourglass architecture at its base 51-class output
         (this is where `model_1427`-style generic pretraining would have
         been used, if you were training from scratch — irrelevant here
         since we're loading an already-fully-trained checkpoint).
      2. Swap in CubiCasa's actual 44-channel final layers.
      3. Load model_best_val_loss_var.pkl into THAT — not model_1427.pth.
    """
    model = get_model('hg_furukawa_original', 51)
    model.conv4_ = torch.nn.Conv2d(256, N_CLASSES, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(N_CLASSES, N_CLASSES, kernel_size=4, stride=4)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    # Their checkpoint dict wraps the actual state dict under 'model_state'
    # (confirmed from train.py's loading code for the --weights argument).
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str, max_dim: int = 512) -> tuple[torch.Tensor, int, int]:
    """
    Reads the image and downscales it BEFORE running it through the network.

    This matters more than it might look: real-world floorplan scans can be
    huge (a 5657x8000 scan -- 45 megapixels -- caused exactly the crash this
    fixes). Running the network at that resolution either OOMs or hits a
    shape mismatch in the encoder-decoder's skip connections, regardless of
    available RAM. CubiCasa5k's own FloorplanSVG loader sidesteps this the
    same way: it runs the network on a downscaled copy (F1_scaled.png), then
    upsamples only the PREDICTION back to the original resolution afterward
    -- it never runs the network at full scan resolution either.

    Returns the ORIGINAL (un-resized) height/width so run_inference() can
    upsample the prediction back to match the source image, even though the
    network itself only ever sees the smaller working resolution.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = image.shape[:2]

    scale = max_dim / max(orig_height, orig_width)
    # Round to a multiple of 32 -- safe for this architecture's downsampling
    # depth (several 2x pooling stages); an odd/arbitrary size is what
    # actually causes skip-connection shape mismatches during upsampling.
    new_height = max(int(round(orig_height * scale / 32)) * 32, 32)
    new_width = max(int(round(orig_width * scale / 32)) * 32, 32)

    resized = cv2.resize(image, (new_width, new_height))
    img = np.moveaxis(resized, -1, 0).astype(np.float32)
    img_tensor = torch.tensor(img).unsqueeze(0)
    # Matches FloorplanSVG.transform(): normalize to [-1, 1]
    img_tensor = 2 * (img_tensor / 255.0) - 1
    return img_tensor, orig_height, orig_width


@torch.no_grad()
def run_inference(model, img_tensor: torch.Tensor, height: int, width: int, device: torch.device):
    img_tensor = img_tensor.to(device)
    pred = model(img_tensor)
    pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)

    # get_polygons() (floortrans.post_prosessing, 2018-era code) calls
    # .numpy() internally and assumes CPU tensors -- move off the GPU here,
    # before split_prediction, rather than deeper in the pipeline where the
    # error is much less obvious ("can't convert cuda:0 device type tensor
    # to numpy" doesn't point at which of the three split tensors caused it).
    pred = pred.cpu()

    heatmaps, rooms, icons = split_prediction(pred, (height, width), SPLIT)
    return heatmaps, rooms, icons


def _debug_inspect_polygons(polygons, types, room_polygons, room_types):
    """
    Run this once against a real image before trusting polygons_to_json()
    below — prints the actual structure so you can confirm/fix the
    assumptions polygons_to_json() makes about how to walk these objects.
    """
    print("=== get_polygons() output inspection ===")
    print(f"polygons: type={type(polygons)}, len={len(polygons) if hasattr(polygons, '__len__') else 'N/A'}")
    if len(polygons) > 0:
        print(f"  polygons[0]: type={type(polygons[0])}, value={polygons[0]}")
    print(f"types: type={type(types)}, value={types[:5] if hasattr(types, '__getitem__') else types}")
    print(f"room_polygons: type={type(room_polygons)}, len={len(room_polygons) if hasattr(room_polygons, '__len__') else 'N/A'}")
    if len(room_polygons) > 0:
        print(f"  room_polygons[0]: type={type(room_polygons[0])}, value={room_polygons[0]}")
    print(f"room_types: type={type(room_types)}, value={room_types[:5] if hasattr(room_types, '__getitem__') else room_types}")
    print("=========================================")


def geometry_to_json(geom) -> list:
    """
    Normalizes ANY shape get_polygons() has been observed to return --
    numpy arrays, plain point lists, or Shapely Polygon/MultiPolygon/
    GeometryCollection objects -- into one consistent structure: a list of
    contours, each a list of [x, y] points.

    Confirmed necessary, not guessed: room_polygons entries come back as
    real Shapely geometries (a MultiPolygon's separate .geoms are exactly
    why some rooms in real output showed multiple disconnected contours,
    e.g. a Bath split across two areas of a floorplan). np.asarray() on a
    Shapely object doesn't do anything sensible, which is what broke before.
    """
    if geom is None:
        return []

    if isinstance(geom, np.ndarray):
        geom = geom.tolist()

    if isinstance(geom, (list, tuple)):
        if len(geom) == 0:
            return []
        first = geom[0]
        # A "point" is a 2-number pair. If geom[0] is itself a point, this
        # is one flat contour -- wrap it. Otherwise it's already a list of
        # contours.
        if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[0], (int, float)):
            return [list(geom)]
        return list(geom)

    if hasattr(geom, "geom_type"):  # Shapely geometry
        if geom.is_empty:
            return []
        if isinstance(geom, Polygon):
            return [list(geom.exterior.coords)]
        if isinstance(geom, MultiPolygon):
            return [list(poly.exterior.coords) for poly in geom.geoms]
        if isinstance(geom, GeometryCollection):
            contours = []
            for sub in geom.geoms:
                contours.extend(geometry_to_json(sub))
            return contours
        return []

    return []


def _extract_class_id(type_info) -> int:
    """
    room_types[i] / types[i] entries can apparently be either a bare int
    class id or a dict wrapping one ({"class": N, ...} or {"label": N, ...})
    -- confirmed necessary from real debugging, not guessed. Handles both
    rather than assuming one.
    """
    if isinstance(type_info, dict):
        return int(type_info.get("class", type_info.get("label", -1)))
    return int(type_info)


def _centroid_of_contours(contours: list) -> list:
    points = [pt for contour in contours for pt in contour]
    if not points:
        return [0.0, 0.0]
    arr = np.asarray(points, dtype=float)
    return arr.mean(axis=0).tolist()


def polygons_to_json(polygons, types, room_polygons, room_types, min_icon_confidence: float = 0.3) -> dict:
    """
    Converts get_polygons()'s output into the {"rooms", "walls", "openings",
    "icons"} JSON schema. `types[i]` is a dict like {"type": "wall", "class":
    2} or {"type": "icon", "class": N, "prob": p} (confirmed from real
    output). Geometry values (room_polygons and, defensively, polygons too)
    may be Shapely objects rather than plain arrays -- geometry_to_json()
    normalizes either.
    """
    result = {"rooms": [], "walls": [], "openings": [], "icons": []}

    for i, poly in enumerate(room_polygons):
        contours = geometry_to_json(poly)
        if i < len(room_types):
            room_class_id = _extract_class_id(room_types[i])
        else:
            room_class_id = None
        room_name = ROOM_CLASSES[room_class_id] if room_class_id is not None and 0 <= room_class_id < len(ROOM_CLASSES) else "Unknown"
        result["rooms"].append({
            "id": f"room_{i}",
            "type": room_name,
            "polygon": contours,
        })

    seen_wall_keys = set()
    seen_icon_keys = set()

    for i, poly in enumerate(polygons):
        poly_type = types[i] if i < len(types) else {}
        contours = geometry_to_json(poly)
        # dedupe key: class + rounded contour coords, since get_polygons()
        # was observed emitting exact duplicate entries for some detections
        rounded = tuple(tuple(tuple(round(c, 1) for c in pt) for pt in contour) for contour in contours)
        dedupe_key = (poly_type.get("type"), poly_type.get("class"), rounded)

        if poly_type.get("type") == "wall":
            if dedupe_key in seen_wall_keys:
                continue
            seen_wall_keys.add(dedupe_key)
            result["walls"].append({
                "id": f"wall_{len(result['walls'])}",
                "polygon": contours,
            })

        elif poly_type.get("type") == "icon":
            prob = float(poly_type.get("prob", 1.0))
            if prob < min_icon_confidence:
                continue  # filter low-confidence noise (observed probs as low as 0.001)
            if dedupe_key in seen_icon_keys:
                continue
            seen_icon_keys.add(dedupe_key)

            class_id = poly_type.get("class")
            icon_name = ICON_CLASSES[class_id] if class_id is not None and class_id < len(ICON_CLASSES) else "Unknown"
            centroid = _centroid_of_contours(contours)
            entry = {
                "id": f"icon_{len(result['icons']) + len(result['openings'])}",
                "type": icon_name,
                "confidence": round(prob, 3),
                "polygon": contours,
                "centroid": centroid,
            }
            # Window/Door are structural openings, not furniture/fixtures --
            # route separately so "openings" isn't empty like before.
            if icon_name in ("Window", "Door"):
                result["openings"].append(entry)
            else:
                result["icons"].append(entry)

        else:
            # Unrecognized type_raw shape -- keep it rather than silently
            # dropping data, but flag it clearly for inspection.
            result.setdefault("unclassified", []).append({
                "id": f"unclassified_{i}",
                "type_raw": poly_type,
                "polygon": contours,
            })

    return _to_json_safe(result)


def _to_json_safe(obj):
    """
    Recursively converts numpy scalars/arrays anywhere in a nested
    dict/list structure into native Python types.

    Why this exists rather than fixing individual call sites: round() on a
    numpy scalar (e.g. np.float32) returns ANOTHER numpy scalar, not a
    Python float -- numpy scalars implement their own __round__. That's
    confirmed to have broken `round(prob, 3)` for the confidence field, but
    get_polygons() could plausibly embed numpy scalars in other places too
    (e.g. inside a plain list rather than a numpy array, which
    geometry_to_json's list-handling branch wouldn't catch). Sanitizing the
    whole structure once, right before it's returned, is more robust than
    chasing each leak individually.
    """
    if isinstance(obj, np.generic):  # any numpy scalar: float32, int64, bool_, etc.
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--weights", type=str, default="model_best_val_loss_var.pkl")
    parser.add_argument("--out", type=str, default="result.json")
    parser.add_argument("--max-dim", type=int, default=512,
                         help="Longest side the image is downscaled to before running the network")
    parser.add_argument("--min-icon-confidence", type=float, default=0.3,
                         help="Filter out icon detections below this confidence")
    parser.add_argument("--inspect", action="store_true",
                         help="Print get_polygons()'s raw output structure instead of writing JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.weights} on {device}...")
    model = load_model(args.weights, device)

    print(f"Running inference on {args.image}...")
    img_tensor, height, width = preprocess_image(args.image, max_dim=args.max_dim)
    heatmaps, rooms, icons = run_inference(model, img_tensor, height, width, device)

    polygons, types, room_polygons, room_types = get_polygons(
        (heatmaps, rooms, icons), threshold=0.2, all_opening_types=[1, 2]  # 1=Window, 2=Door
    )

    if args.inspect:
        _debug_inspect_polygons(polygons, types, room_polygons, room_types)
        return

    result = polygons_to_json(polygons, types, room_polygons, room_types, min_icon_confidence=args.min_icon_confidence)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out} ({len(result['rooms'])} rooms, {len(result['walls'])} walls, "
          f"{len(result['openings'])} openings, {len(result['icons'])} icons)")


if __name__ == "__main__":
    main()