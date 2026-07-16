import json
from pathlib import Path

import modal

app = modal.App("cubicasa5k-original-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # needed by opencv
    .pip_install(
        "torch",
        "opencv-python-headless",
        "numpy",
        "scipy",
        "scikit-image",
        "svgpathtools",
        "shapely",
        "matplotlib",
        "lmdb",
    )
    .add_local_dir("floortrans", remote_path="/app/floortrans")
    .add_local_file("run_inference_to_json.py", remote_path="/app/run_inference_to_json.py")
)

volume = modal.Volume.from_name("cubicasa5k-io", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume},
    timeout=600,
)
def run_inference_remote(
    image_filename: str,
    output_filename: str = "result.json",
    weights_filename: str = "model_best_val_loss_var.pkl",
):
    import sys
    import json

    sys.path.insert(0, "/app")

    import torch
    from floortrans.post_prosessing import get_polygons
    from run_inference_to_json import (
        load_model,
        preprocess_image,
        run_inference as _run_inference,
        polygons_to_json,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(f"/data/{weights_filename}", device)
    img_tensor, height, width = preprocess_image(f"/data/{image_filename}")
    heatmaps, rooms, icons = _run_inference(model, img_tensor, height, width, device)

    polygons, types, room_polygons, room_types = get_polygons(
        (heatmaps, rooms, icons),
        threshold=0.2,
        all_opening_types=[1, 2],
    )

    result = polygons_to_json(polygons, types, room_polygons, room_types)

    import numpy as np

    def json_converter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(f"/data/{output_filename}", "w") as f:
        json.dump(result, f, indent=2, default=json_converter)
    volume.commit()
    return output_filename


@app.local_entrypoint()
def main(image_path: str, out: str = "result.json"):
    filename = Path(image_path).name
    with volume.batch_upload() as batch:
        batch.put_file(image_path, f"/{filename}")
    output_name = Path(out).name
    run_inference_remote.remote(filename, output_name)

    print(f"Saved output to Modal Volume: /data/{output_name}")