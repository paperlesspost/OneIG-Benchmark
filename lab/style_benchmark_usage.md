### OneIG Style Benchmark on Modal

Run the OneIG-Benchmark style evaluation on Modal using a GPU, with persistent volumes for inputs, models, and results.

#### Script
- Entry: `modal/apps/OneIG-Benchmark/style_benchmark_modal.py`
- App name: `oneig-style-benchmark`
- Mounts:
  - Repo mounted at `/oneig`
  - Volume `style-embeddings` mounted at `/vol`
  - Volume `benchmarks` mounted at `/benchmarks`
- Paths used:
  - Images: `/vol/test-imgs3/<model_name>/...`
  - CSD checkpoint: `/oneig/scripts/style/models/checkpoint.pth` (persisted on `/vol`)
  - Results: `/benchmarks/<model_name>/scores/style_score_<MODE>_*.csv`

---

### Prerequisites
- Modal CLI installed and logged in (`pip install modal`, `modal token new`).
- This repo checked out with OneIG-Benchmark present under `modal/apps/OneIG-Benchmark/`.

---

### Prepare inputs (one-time)
1) Create or reuse the volume (auto-created on first run). Upload your images so each model has its own folder:
   ```bash
   # Example for a model named "doubao"
   modal volume put style-embeddings ./local-images/doubao /vol/test-imgs3/doubao
   # You can repeat for more models:
   modal volume put style-embeddings ./local-images/imagen4 /vol/test-imgs3/imagen4
   ```

2) Filename format and grid size
   - Filenames should begin with a 3-digit id (e.g., `000.webp`, `001.jpg`, ...).
   - Grid setting `2` means each file is a 2x2 grid image; use `1` if each file is a single image.

3) (Optional) Download the CSD checkpoint ahead of time
   ```bash
   modal call modal/apps/OneIG-Benchmark/style_benchmark_modal.py::download_csd_checkpoint
   ```

---

### Run the benchmark
Basic single-model run (2x2 grid images):
```bash
modal run modal/apps/OneIG-Benchmark/style_benchmark_modal.py::main \
  --mode EN \
  --model-names-csv doubao \
  --image-grid-csv 2
```

Multiple models:
```bash
modal run modal/apps/OneIG-Benchmark/style_benchmark_modal.py::main \
  --mode EN \
  --model-names-csv doubao,imagen4 \
  --image-grid-csv 2,2
```

Chinese prompts mode:
```bash
modal run modal/apps/OneIG-Benchmark/style_benchmark_modal.py::main --mode ZH \
  --model-names-csv doubao --image-grid-csv 2
```

Notes
- GPU: H200 is requested by default.
- Timeout: 20 minutes per run container.
- The first run will build the image; subsequent runs will be faster.

---

### Retrieve results
Results are saved per model to the `benchmarks` volume under `/benchmarks/<model_name>/scores/`.

Single model example (`doubao`):
```bash
modal volume get benchmarks /benchmarks/doubao/scores/style_score_EN_*.csv ./results/
```

Multiple models: repeat the command for each model name.

---

### Advanced usage
- Custom images directory (remote/local path):
  ```bash
  modal call modal/apps/OneIG-Benchmark/style_benchmark_modal.py::run_style \
    --mode EN \
    --image-dir-remote /vol/test-imgs3 \
    --model-names '["doubao","imagen4"]' \
    --image-grid '[2,2]'
  ```
- Deps and GPU settings live in `style_benchmark_modal.py`. Adjust pins or GPU in the decorators if needed.

---

### Layout reference
- `/oneig` → repository mount (contains `scripts/`, `results/`, etc.)
- `/vol` → Modal Volume `style-embeddings` (inputs and outputs)
  - Images: `/vol/test-imgs3/<model_name>/...`
  - Models cache (CSD): `/oneig/scripts/style/models/checkpoint.pth`
 - `/benchmarks` → Modal Volume `benchmarks` (shared with the style embedding converter)
   - Scores per model: `/benchmarks/<model_name>/scores/style_score_<MODE>_*.csv`

---

### Troubleshooting
- No images found: ensure you uploaded to `/vol/test-imgs3/<model_name>/` and the mount target is correct in the `modal volume put` command.
- Wrong scores or missing prompts: check filenames begin with the 3-digit id (`000`, `001`, ...), and grid size matches your files.
- Slow downloads on first run: model weights and tooling are cached via volumes and HF cache; later runs are faster.


