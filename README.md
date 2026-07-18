# FloorPlanParser — FastAPI Service

A FastAPI service that wraps the CubiCasa5k pretrained model to parse floorplan
images into structured JSON (rooms, walls, openings, icons) and to render
overlay visualizations.

> This README covers **only the FastAPI service** (`app_fastapi.py`). The
> training/eval scripts are out of scope here.

## Requirements

- Run from inside a CubiCasa5k repo clone — the service imports `floortrans`
  and `run_inference_to_json`, so the working directory must contain them.
- **CUDA GPU required.** The model is loaded onto `cuda` at startup and the
  service fails loudly (rather than silently falling back to CPU) if no GPU is
  available.
- `model_best_val_loss_var.pkl` (the weights) in the working directory.

## Install

```bash
pip install fastapi uvicorn python-multipart
```

(`python-multipart` is needed for FastAPI's `File()`/`Form()` uploads.)

## Run locally

```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```

The model loads once at startup (lifespan event). Wait for the `Model loaded.
Ready.` log before sending requests.

## Endpoints

### `GET /health`

Liveness/readiness check.

```bash
curl http://localhost:8000/health
# {"status":"ok","cuda":true,"model_loaded":true}
```

### `POST /parsetojson`

Upload a floorplan image and get back the parsed
`{"rooms","walls","openings","icons"}` JSON.

| Field                | Type   | Default | Description                          |
|----------------------|--------|---------|--------------------------------------|
| `image`              | file   | —       | Floorplan image (required)           |
| `max_dim`            | int    | `512`   | Longest side is resized to this      |
| `min_icon_confidence`| float  | `0.3`   | Drop icons below this confidence     |

```bash
curl -F "image=@floorplan.png" \
     -F "max_dim=512" \
     -F "min_icon_confidence=0.3" \
     http://localhost:8000/parsetojson -o result.json
```

### `POST /visualize`

Render a PNG overlay. Accepts an image, JSON, or both:

| Input                 | Behavior                                                  |
|-----------------------|-----------------------------------------------------------|
| `image` only          | Runs inference, overlays the result on the image          |
| `json_data` only      | Draws the given result on a blank canvas                  |
| `image` + `json_data` | Overlays the (already-computed) JSON on the image; **no** inference re-run |
| neither               | `400`                                                     |

| Field                | Type   | Default | Description                          |
|----------------------|--------|---------|--------------------------------------|
| `image`              | file   | —       | Optional background image            |
| `json_data`          | string | —       | Optional JSON result (as a string)   |
| `max_dim`            | int    | `512`   | Longest side resized to this         |
| `min_icon_confidence`| float  | `0.3`   | Icon confidence threshold            |

```bash
# Image only -> inference + overlay
curl -F "image=@floorplan.png" http://localhost:8000/visualize -o viz.png

# JSON only -> drawn on blank canvas
curl -F "json_data=$(cat result.json)" http://localhost:8000/visualize -o viz.png

# Image + precomputed JSON (no re-run)
curl -F "image=@floorplan.png" -F "json_data=$(cat result.json)" \
     http://localhost:8000/visualize -o viz.png
```

## Python client example

```python
import requests

with open("floorplan.png", "rb") as f:
    r = requests.post(
        "http://localhost:8000/parsetojson",
        files={"image": f},
        data={"max_dim": 512, "min_icon_confidence": 0.3},
    )
result = r.json()
print(result["rooms"])
```

## Deploy on Modal (optional)

`modal_app_fastapi.py` is a thin deployment wrapper (no Modal code lives in
`app_fastapi.py`, so it still runs under plain uvicorn). It builds the image,
provisions a GPU, and mounts the weights volume.

```bash
modal volume create cubicasa5k-io
modal volume put cubicasa5k-io model_best_val_loss_var.pkl /model_best_val_loss_var.pkl

modal deploy modal_app_fastapi.py   # persistent, public URL
# or for local dev with hot-reload:
modal serve modal_app_fastapi.py
```

POST to `<url>/parsetojson` and `<url>/visualize` exactly as above.
