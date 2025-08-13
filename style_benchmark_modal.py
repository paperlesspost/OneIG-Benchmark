from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import modal


# Constants
MINUTES = 60  # seconds


# Resolve local repo directory (this file lives inside the repo root)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_REMOTE = "/oneig"

# Optional: mount a local image directory if IMAGE_DIR_LOCAL is set
_local_images_dir = os.environ.get("IMAGE_DIR_LOCAL")
IMAGES_REMOTE = "/images" if _local_images_dir else None


# Volumes
hf_cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
# Reuse the same volume layout as modal_style_benchmark.py
style_volume = modal.Volume.from_name("style-embeddings", create_if_missing=True)
# Shared benchmarks volume used by style_embedding_converter.py
bench_volume = modal.Volume.from_name("benchmarks", create_if_missing=True)


# Image with pinned requirements for OneIG-Bench
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "wget", "git")
    .pip_install(
        # Core pins from OneIG-Bench
        "torch==2.6.0",
        "torchvision==0.21.0",
        "triton==3.2.0",
        "transformers==4.50.0",
        "accelerate==1.4.0",
        "dreamsim",
        "openai-clip",
        "qwen_vl_utils",
        "peft",
        "pandas",
        "megfile",
        # Utilities used by the scripts
        "pillow",
        "tqdm",
        "gdown",
        # HF fast downloads
        "huggingface_hub[hf_transfer]==0.28.1",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": "/root/.cache/huggingface",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

# Mount local repo and optional local images directory via the Image API (Modal 1.0+)
image = image.add_local_dir(str(CURRENT_DIR), remote_path=REPO_REMOTE)
if _local_images_dir:
    image = image.add_local_dir(_local_images_dir, remote_path=str(IMAGES_REMOTE))


app = modal.App("oneig-style-benchmark", image=image)


def _as_str_list(values: Iterable) -> list[str]:
    return [str(v) for v in values]


@app.function(
    image=image,
    volumes={
        "/vol": style_volume,
    },
    timeout=20 * MINUTES,
)
def download_csd_checkpoint(force_redownload: bool = False) -> str:
    """Download the CSD checkpoint to scripts/style/models/checkpoint.pth.

    The file is stored in a persistent Volume mounted at
    /oneig/scripts/style/models so subsequent runs don't need to re-download.
    """
    import os

    # Persist checkpoint inside the shared volume
    target_dir = "/vol/scripts/style/models"
    os.makedirs(target_dir, exist_ok=True)
    target_path = f"{target_dir}/checkpoint.pth"

    if not force_redownload and os.path.exists(target_path):
        print(f"✅ CSD checkpoint already present: {target_path}")
        return target_path

    # Google Drive file id from README
    file_id = "1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46"

    import gdown

    print("⬇️  Downloading CSD checkpoint (this can take a while)...")
    gdown.download(id=file_id, output=target_path, quiet=False)

    # Ensure CLIP ViT-L-14 weights are available at the expected path
    try:
        import clip as _clip
        # This will download to target_dir as ViT-L-14.pt if missing
        _clip.load("ViT-L/14", download_root=target_dir)
        # Ensure it's also present in the repo path for style_score
        clip_src = f"{target_dir}/ViT-L-14.pt"
        clip_dst_dir = f"{REPO_REMOTE}/scripts/style/models"
        clip_dst = f"{clip_dst_dir}/ViT-L-14.pt"
        os.makedirs(clip_dst_dir, exist_ok=True)
        if os.path.exists(clip_src) and not os.path.exists(clip_dst):
            try:
                os.symlink(clip_src, clip_dst)
            except Exception:
                import shutil as _sh
                _sh.copy(clip_src, clip_dst)
    except Exception as _e:
        print(f"⚠️ Could not prefetch CLIP ViT-L/14 weights: {_e}")

    # Make sure the write is visible to other containers immediately
    style_volume.commit()
    print(f"✅ Downloaded CSD checkpoint to: {target_path}")
    return target_path


@app.function(
    image=image,
    gpu="H200",
    timeout=20 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/vol": style_volume,  # expose /vol/test-imgs3 and /vol/results
        "/benchmarks": bench_volume,  # save final scores alongside embeddings
    },
)
def run_style(
    mode: str = "EN",
    image_dir_remote: str | None = "/vol/test-imgs3",
    model_names: str = "[\"doubao\"]",
    image_grid: str = "[2]",
) -> str:
    """Run style benchmark.

    - mode: "EN" or "ZH"
    - image_dir_remote: container path to the images root. Must contain 'anime/'.
      If IMAGE_DIR_LOCAL is set when launching `modal run`, a local directory is
      mounted at /images by default and used here.
    - model_names: list of subfolder names under image_dir_remote/anime
    - image_grid: list of N per model (1->1, 2->2x2, etc.)
    """
    import os
    import subprocess

    if image_dir_remote is None:
        raise RuntimeError(
            "image_dir_remote is None. Set IMAGE_DIR_LOCAL to mount a local images "
            "directory, or pass a container path explicitly."
        )

    cwd = REPO_REMOTE
    os.makedirs(f"{cwd}/results", exist_ok=True)
    os.makedirs("/vol/results", exist_ok=True)

    # Ensure the CSD checkpoint exists in the shared volume
    _ = download_csd_checkpoint.remote()

    # Link or copy the checkpoint into the repo path expected by style_score
    vol_ckpt = "/vol/scripts/style/models/checkpoint.pth"
    repo_models_dir = f"{cwd}/scripts/style/models"
    repo_ckpt = f"{repo_models_dir}/checkpoint.pth"
    os.makedirs(repo_models_dir, exist_ok=True)
    try:
        if os.path.exists(vol_ckpt) and not os.path.exists(repo_ckpt):
            os.symlink(vol_ckpt, repo_ckpt)
    except Exception:
        # Fallback to copying if symlinks are not allowed
        if os.path.exists(vol_ckpt):
            import shutil as _sh
            _sh.copy(vol_ckpt, repo_ckpt)

    # Ensure CLIP weights are present where CSD_config expects them
    vol_clip = "/vol/scripts/style/models/ViT-L-14.pt"
    repo_clip = f"{repo_models_dir}/ViT-L-14.pt"
    try:
        if os.path.exists(vol_clip) and not os.path.exists(repo_clip):
            os.symlink(vol_clip, repo_clip)
    except Exception:
        if os.path.exists(vol_clip):
            import shutil as _sh
            _sh.copy(vol_clip, repo_clip)

    # Normalize CLI inputs (accept JSON strings or Python lists)
    try:
        import json as _json
        model_names_list = (
            _json.loads(model_names) if isinstance(model_names, str) else list(model_names)
        )
        image_grid_list = (
            _json.loads(image_grid) if isinstance(image_grid, str) else list(image_grid)
        )
    except Exception as _e:
        raise RuntimeError(f"Invalid model_names/image_grid: {model_names}, {image_grid}: {_e}")

    # If user-provided embeddings are present in /vol/embeddings, copy them
    # to the paths expected by scripts.style.style_score
    embeds_src_dir = "/vol/embeddings"
    embeds_dst_dir = f"{cwd}/scripts/style"
    try:
        if os.path.exists(f"{embeds_src_dir}/CSD_embed.pt"):
            os.makedirs(embeds_dst_dir, exist_ok=True)
            import shutil as _sh
            _sh.copy(f"{embeds_src_dir}/CSD_embed.pt", f"{embeds_dst_dir}/CSD_embed.pt")
            print("✔ Using custom CSD_embed.pt from /vol/embeddings")
        if os.path.exists(f"{embeds_src_dir}/SE_embed.pt"):
            os.makedirs(embeds_dst_dir, exist_ok=True)
            import shutil as _sh
            _sh.copy(f"{embeds_src_dir}/SE_embed.pt", f"{embeds_dst_dir}/SE_embed.pt")
            print("✔ Using custom SE_embed.pt from /vol/embeddings")
        if os.path.exists(f"{embeds_src_dir}/style.csv"):
            os.makedirs(embeds_dst_dir, exist_ok=True)
            import shutil as _sh
            _sh.copy(f"{embeds_src_dir}/style.csv", f"{embeds_dst_dir}/style.csv")
            print("✔ Using custom style.csv from /vol/embeddings")
    except Exception as e:
        print(f"⚠️ Could not copy custom embeddings/style.csv: {e}")

    cmd = [
        "python",
        "-m",
        "scripts.style.style_score",
        "--mode",
        mode,
        "--image_dirname",
        f"{image_dir_remote}",
        "--model_names",
        *[str(x) for x in model_names_list],
        "--image_grid",
        *[str(x) for x in image_grid_list],
    ]

    print("🚀 Running:")
    print(" ", " ".join(cmd))

    # Run inside repo so relative paths resolve (scripts/, results/, etc.)
    subprocess.run(cmd, check=True, cwd=cwd)

    # Copy results into the shared /benchmarks/<model>/scores path and commit
    import glob
    import shutil

    results = glob.glob(f"{cwd}/results/style_score_{mode}_*.csv")
    if results:
        for model_name in model_names_list:
            out_dir = f"/benchmarks/{model_name}/scores"
            os.makedirs(out_dir, exist_ok=True)
            for src in results:
                shutil.copy(src, out_dir)
        # Persist changes to the benchmarks volume
        bench_volume.commit()
    else:
        print("⚠️ No results CSVs found to copy.")

    # Also persist any changes to the style volume (not strictly needed, but safe)
    style_volume.commit()

    # Return a helpful glob path (for single-model runs); print all destinations
    for model_name in model_names_list:
        print(f"📄 Scores saved to: /benchmarks/{model_name}/scores/")
    if len(model_names_list) == 1:
        return f"/benchmarks/{model_names_list[0]}/scores/style_score_{mode}_*.csv"
    # Fallback when multiple models are provided
    return ", ".join(
        [f"/benchmarks/{mn}/scores/style_score_{mode}_*.csv" for mn in model_names_list]
    )


@app.local_entrypoint()
def main(
    mode: str = "EN",
    model_names_csv: str = "doubao",
    image_grid_csv: str = "2",
    ensure_models: bool = True,
) -> None:
    """Local entrypoint to run the style benchmark remotely.

    Usage (with local images):
      IMAGE_DIR_LOCAL=/absolute/path/to/images \
      modal run modal/apps/OneIG-Benchmark/style_benchmark_modal.py::main \
        --mode EN \
        --model-names-csv doubao,imagen4 \
        --image-grid-csv 2,2

    Your images directory must follow the README structure, e.g.:
      /path/to/images/anime/<model_name>/{000.webp,001.webp,...}
    """
    model_names = [s for s in model_names_csv.split(",") if s]
    image_grid = [int(x) for x in image_grid_csv.split(",") if x]

    if ensure_models:
        download_csd_checkpoint.remote()

    out = run_style.remote(
        mode=mode,
        image_dir_remote="/vol/test-imgs3",
        model_names=model_names,
        image_grid=image_grid,
    )
    print(f"✅ Submitted. Results CSV path (glob): {out}")


