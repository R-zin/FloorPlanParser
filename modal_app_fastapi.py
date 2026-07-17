"""
Deploys app_fastapi.py (the CubiCasa5k FastAPI service) on Modal.

Setup:
    modal volume create cubicasa5k-io
    modal volume put cubicasa5k-io model_best_val_loss_var.pkl /model_best_val_loss_var.pkl

Deploy (persistent, gets a public URL):
    modal deploy modal_app_fastapi.py

Or for local dev with hot-reload:
    modal serve modal_app_fastapi.py

Both commands print the served URL -- POST to <url>/parsetojson and
<url>/visualize exactly as documented in app_fastapi.py.

Why this file exists separately from app_fastapi.py: the FastAPI app itself
has no Modal-specific code in it (so it stays runnable locally with plain
uvicorn too, per its own docstring). This file is purely the Modal
packaging/deployment layer around it -- image build, GPU, volume mount, and
pointing MODEL_STATE at the volume's weights path before the app's lifespan
startup loads them.
"""

import modal

app = modal.App("cubicasa5k-fastapi")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "opencv-python-headless",
        "numpy",
        "scipy",
        "scikit-image",
        "svgpathtools",
        "shapely",
        "matplotlib",
        "lmdb",  # floortrans.loaders imports this at module level even if unused
        "fastapi",
        "uvicorn",
        "python-multipart",  # required by FastAPI's File()/Form()
        "pillow",
    )
    .add_local_dir("floortrans", remote_path="/root/floortrans")
    .add_local_file("run_inference_to_json.py", remote_path="/root/run_inference_to_json.py")
    .add_local_file("app_fastapi.py", remote_path="/root/app_fastapi.py")
)

# Same volume from earlier -- holds model_best_val_loss_var.pkl.
io_volume = modal.Volume.from_name("cubicasa5k-io", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/io": io_volume},
    timeout=600,          # per-request timeout
    scaledown_window=600,  # keep the container (and loaded model) warm for 10 min after last request
    min_containers=0,      # set to 1 if you want zero cold-start latency at the cost of idle GPU billing
)
# No @modal.concurrent() needed -- Modal's default is already 1 input at a
# time per container, which is what we want here anyway: one inference call
# uses the whole GPU, so there's no benefit to overlapping requests on the
# same container the way there would be for I/O-bound work.
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, "/root")
    import app_fastapi

    # Point the app at the volume-mounted weights instead of the local-dev
    # default relative path -- must happen before the ASGI lifespan startup
    # event fires and actually loads the model.
    app_fastapi.MODEL_STATE["weights_path"] = "/io/model_best_val_loss_var.pkl"
    return app_fastapi.app
