from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Union

import modal


# Constants
MINUTES = 60  # seconds


# Resolve local repo directory (this file lives inside the repo root)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_REMOTE = "/oneig"

# Images are stored in a Modal Volume mounted at /images
IMAGES_REMOTE = "/images"


# Volumes
hf_cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
bench_volume = modal.Volume.from_name("benchmarks", create_if_missing=True)
images_volume = modal.Volume.from_name("images")
models_volume = modal.Volume.from_name("OneIG-Benchmark", create_if_missing=True)


# NVIDIA CUDA image configuration
cuda_version = "11.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.10"
).entrypoint([])

# Image with pinned requirements for OneIG-Bench
image = (
    cuda_dev_image
    .apt_install("curl", "wget", "git")
    .pip_install(
        # Dependencies from requirements.txt
        # We test our benchmark using torch==2.6.0, torchvision==0.21.0 with cuda-11.8, python==3.10.
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers",
        "triton==3.2.0",
        "accelerate==1.4.0",
        "dreamsim",
        "openai-clip",
        "qwen_vl_utils",
        "peft",
        "pandas",
        "megfile",
        "huggingface_hub[hf_transfer]==0.28.1",
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": "/root/.cache/huggingface",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

# Mount local repo via the Image API (Modal 1.0+)
image = image.add_local_dir(str(CURRENT_DIR), remote_path=REPO_REMOTE)


app = modal.App("OneIG-Benchmark", image=image)


def _as_str_list(values: Iterable) -> list[str]:
    return [str(v) for v in values]


def _copy_results_to_bench(volume: modal.Volume, repo_results_glob: str, model_name: str) -> None:
    import glob
    import shutil
    results = glob.glob(repo_results_glob)
    if not results:
        print("⚠️ No results found for glob:", repo_results_glob)
        return
    out_dir = f"/benchmarks/{model_name}/scores"
    os.makedirs(out_dir, exist_ok=True)
    for src in results:
        shutil.copy(src, out_dir)
    volume.commit()


# For us run_all is alignment, diversity, and text. This because we are testing LoRA models that are created to do specific styles.
@app.function(
    image=image,
    gpu="H100",
    timeout=240 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/benchmarks": bench_volume,
        "/images": images_volume,
        "/oneig/models": models_volume,
    },
    max_containers=5,
)
def run_all(
    mode: str = "EN",
    model_name: str = "",
    image_grid: str = "[2,2]",
    image_dir_remote: str | None = None,
) -> dict[str, str]:
    """Run alignment, diversity, and text with one shared config.

    Returns a dict of result glob paths keyed by test name.
    """
    if not model_name:
        raise RuntimeError("no model name specified")

    if image_dir_remote is None:
        image_dir_remote = str(IMAGES_REMOTE or "")
    if not image_dir_remote:
        raise RuntimeError("image_dir_remote is empty. Set IMAGE_DIR_LOCAL or pass a path.")

    # Normalize inputs
    import json as _json
    ig_list = _json.loads(image_grid) if isinstance(image_grid, str) else list(image_grid)

    # Run functions directly on the same GPU
    align_out = run_alignment.local(
        mode=mode,
        image_dir_remote=image_dir_remote,
        model_name=model_name,
        image_grid=ig_list,
    )
    div_out = run_diversity.local(
        mode=mode,
        image_dir_remote=image_dir_remote,
        model_name=model_name,
        image_grid=ig_list,
    )
    text_out = run_text.local(
        mode=mode,
        image_dir_remote_text=f"{image_dir_remote}/text",
        model_name=model_name,
        image_grid=ig_list,
    )

    print( {"alignment": align_out, "diversity": div_out, "text": text_out} )
    return {
        "alignment": align_out,
        "diversity": div_out,
        "text": text_out,
    }

@app.function(
    image=image,
    gpu="H100",
    timeout=120 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/benchmarks": bench_volume,
        "/images": images_volume,
        "/oneig/models": models_volume,
    },
    max_containers=5,
    scaledown_window= 15,
)
def run_alignment(
    mode: str = "EN",
    image_dir_remote: str | None = None,
    model_name: str = "",
    image_grid: str = "[2,2]",
    class_items: list[str] | None = ["human", "object"],
) -> str:
    """Run alignment benchmark for human and object only.

    - mode: "EN" or "ZH"
    - image_dir_remote: container path to the images root. Must contain 'human/' and 'object/'.
    - model_name: subfolder name under each class
    - image_grid: list of N per model (1->1, 2->2x2, etc.)
    - class_items: list of class items to benchmark (default: ["human", "object"])
    """
    import os
    import subprocess

    if not model_name:
        raise RuntimeError("no model name specified")

    if image_dir_remote is None:
        image_dir_remote = str(IMAGES_REMOTE or "")
    if not image_dir_remote:
        raise RuntimeError("image_dir_remote is empty. Set IMAGE_DIR_LOCAL or pass a path.")

    cwd = REPO_REMOTE
    os.makedirs(f"{cwd}/results", exist_ok=True)

    # Normalize CLI inputs (accept JSON strings or Python lists)
    try:
        import json as _json
        image_grid_list = (
            _json.loads(image_grid) if isinstance(image_grid, str) else list(image_grid)
        )
    except Exception as _e:
        raise RuntimeError(f"Invalid image_grid: {image_grid}: {_e}")

    cmd = [
        "python",
        "-m",
        "scripts.alignment.alignment_score",
        "--mode",
        mode,
        "--image_dirname",
        f"{image_dir_remote}",
        "--model_names",
        model_name,
        "--image_grid",
        *[str(x) for x in image_grid_list],
        "--class_items",
        *class_items,
    ]

    print("🚀 Running:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

    _copy_results_to_bench(bench_volume, f"{cwd}/results/alignment_score_{mode}_*.csv", model_name)
    print(f"📄 Alignment scores saved to: /benchmarks/{model_name}/scores/")
    return f"/benchmarks/{model_name}/scores/alignment_score_{mode}_*.csv"


@app.function(
    image=image,
    gpu="H100",
    timeout=120 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/benchmarks": bench_volume,
        "/images": images_volume,
        "/oneig/models": models_volume,
    },
    max_containers=5,
)
def run_diversity(
    mode: str = "EN",
    image_dir_remote: str | None = None,
    model_name: str = "",
    image_grid: str = "[2]",
    class_items: list[str] | None = ["human", "object", "text"],
) -> str:
    """Run diversity benchmark for human, object, and text only.
    
    - class_items: list of class items to benchmark (default: ["human", "object", "text"])
    """
    import os
    import subprocess

    if not model_name:
        raise RuntimeError("no model name specified")

    if image_dir_remote is None:
        image_dir_remote = str(IMAGES_REMOTE or "")
    if not image_dir_remote:
        raise RuntimeError("image_dir_remote is empty. Set IMAGE_DIR_LOCAL or pass a path.")

    cwd = REPO_REMOTE
    os.makedirs(f"{cwd}/results", exist_ok=True)

    try:
        import json as _json
        image_grid_list = (
            _json.loads(image_grid) if isinstance(image_grid, str) else list(image_grid)
        )
    except Exception as _e:
        raise RuntimeError(f"Invalid image_grid: {image_grid}: {_e}")

    cmd = [
        "python",
        "-m",
        "scripts.diversity.diversity_score",
        "--mode",
        mode,
        "--image_dirname",
        f"{image_dir_remote}",
        "--model_names",
        model_name,
        "--image_grid",
        *[str(x) for x in image_grid_list],
        "--class_items",
        *class_items,
    ]

    print("🚀 Running:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

    _copy_results_to_bench(bench_volume, f"{cwd}/results/diversity_score_{mode}_*.csv", model_name)
    print(f"📄 Diversity scores saved to: /benchmarks/{model_name}/scores/")
    return f"/benchmarks/{model_name}/scores/diversity_score_{mode}_*.csv"


@app.function(
    image=image,
    gpu="H100",
    timeout=120 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/benchmarks": bench_volume,
        "/images": images_volume,
        "/oneig/models": models_volume,
    },
    max_containers=5,
)
def run_text(
    mode: str = "EN",
    image_dir_remote_text: str | None = None,
    model_name: str = "",
    image_grid: str = "[2,2]",
) -> str:
    """Run text benchmark (expects images under <image_dir_remote_text>/<model_name>/id.webp)."""
    import os
    import subprocess

    if not model_name:
        raise RuntimeError("no model name specified")

    if image_dir_remote_text is None:
        # Text script expects images in a subdir; our sample script saves text under /images/text/
        base = str(IMAGES_REMOTE or "")
        image_dir_remote_text = f"{base}/text" if base else ""
    if not image_dir_remote_text:
        raise RuntimeError(
            "image_dir_remote_text is empty. Set IMAGE_DIR_LOCAL or pass a path."
        )

    cwd = REPO_REMOTE
    os.makedirs(f"{cwd}/results", exist_ok=True)

    try:
        import json as _json
        image_grid_list = (
            _json.loads(image_grid) if isinstance(image_grid, str) else list(image_grid)
        )
    except Exception as _e:
        raise RuntimeError(f"Invalid image_grid: {image_grid}: {_e}")

    cmd = [
        "python",
        "-m",
        "scripts.text.text_score",
        "--mode",
        mode,
        "--image_dirname",
        f"{image_dir_remote_text}",
        "--model_names",
        model_name,
        "--image_grid",
        *[str(x) for x in image_grid_list],
    ]

    print("🚀 Running:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

    _copy_results_to_bench(bench_volume, f"{cwd}/results/text_score_{mode}_*.csv", model_name)
    print(f"📄 Text scores saved to: /benchmarks/{model_name}/scores/")
    return f"/benchmarks/{model_name}/scores/text_score_{mode}_*.csv"
