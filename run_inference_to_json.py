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
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

import cv2
import numpy as np
import torch
import torch.nn.functional as F

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


def preprocess_image(image_path: str) -> tuple[torch.Tensor, int, int]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    img = np.moveaxis(image, -1, 0).astype(np.float32)
    img_tensor = torch.tensor(img).unsqueeze(0)
    # Matches FloorplanSVG.transform(): normalize to [-1, 1]
    img_tensor = 2 * (img_tensor / 255.0) - 1
    return img_tensor, height, width


@torch.no_grad()
def run_inference(model, img_tensor: torch.Tensor, height: int, width: int, device: torch.device):
    img_tensor = img_tensor.to(device)
    pred = model(img_tensor)
    pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)

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


def polygons_to_json(polygons, types, room_polygons, room_types) -> dict:
    """
    Assumed structure (verify this):
      - room_polygons[i]: array-like of (x, y) points for room i's polygon
      - room_types[i]: int class id into ROOM_CLASSES for room i
      - polygons[i]: array-like of (x, y) points for a non-room polygon
        (walls / openings)
      - types[i]: some (class, subtype) or class id identifying what
        polygons[i] represents — walls vs. windows vs. doors
    """
    result = {"rooms": [], "walls": [], "openings": []}

    def geometry_to_json(geom):

        if isinstance(geom, np.ndarray):
            return geom.tolist()

        if not hasattr(geom, "geom_type"):
            return geom

        if geom.is_empty:
            return []


        if isinstance(geom, Polygon):
            return [list(geom.exterior.coords)]

        if isinstance(geom, MultiPolygon):
            return [list(poly.exterior.coords) for poly in geom.geoms]

        if isinstance(geom, GeometryCollection):
            polygons = []
            for g in geom.geoms:
                polygons.extend(geometry_to_json(g))
            return polygons

        return []

    print("room_types:", room_types)
    if room_types:
        print("first room_type:", room_types[0])
        print("first room_type type:", type(room_types[0]))

    for i, poly in enumerate(room_polygons):
        if i < len(room_types):
            room_info = room_types[i]
            if isinstance(room_info, dict):
                room_class_id = int(room_info.get("class", room_info.get("label", -1)))
            else:
                room_class_id = int(room_info)
        else:
            room_class_id = None
        room_name = ROOM_CLASSES[room_class_id] if room_class_id is not None and room_class_id < len(ROOM_CLASSES) else "Unknown"
        result["rooms"].append({
            "id": f"room_{i}",
            "type": room_name,
            "polygon": geometry_to_json(poly),
        })

    for i, poly in enumerate(polygons):
        poly_type = types[i] if i < len(types) else None
        entry = {
            "id": f"polygon_{i}",
            "type_raw": poly_type if not hasattr(poly_type, "tolist") else poly_type.tolist(),
            "polygon": geometry_to_json(poly),
        }
        result["walls"].append(entry)
    print(room_polygons[0].geom_type)
    print(room_polygons)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--weights", type=str, default="model_best_val_loss_var.pkl")
    parser.add_argument("--out", type=str, default="result.json")
    parser.add_argument("--inspect", action="store_true",
                         help="Print get_polygons()'s raw output structure instead of writing JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.weights} on {device}...")
    model = load_model(args.weights, device)

    print(f"Running inference on {args.image}...")
    img_tensor, height, width = preprocess_image(args.image)
    heatmaps, rooms, icons = run_inference(model, img_tensor, height, width, device)

    polygons, types, room_polygons, room_types = get_polygons(
        (heatmaps, rooms, icons), threshold=0.2, all_opening_types=[1, 2]  # 1=Window, 2=Door
    )

    if args.inspect:
        _debug_inspect_polygons(polygons, types, room_polygons, room_types)
        return

    result = polygons_to_json(polygons, types, room_polygons, room_types)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out} ({len(result['rooms'])} rooms, {len(result['walls'])} wall/opening polygons)")


if __name__ == "__main__":
    main()