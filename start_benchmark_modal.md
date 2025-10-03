### ONEIG-Benchmark on Modal: Alignment, Diversity, Text

This guide shows how to run the OneIG-Bench alignment, diversity, and text evaluations on Modal using the app defined in `start_benchmark_modal.py`.

### Prerequisites
- Modal CLI installed and logged in.
- Local images generated for these classes only: `human`, `object`, `text`.
- Images should be located in the /images modal.com volume
```
/images
├── human/
│   ├── gpt-4o/
│   │   ├── 000.webp
│   │   └── ...
│   └── imagen4/
├── object/
│   ├── gpt-4o/
│   └── imagen4/
└── text/
    ├── gpt-4o/
    └── imagen4/
```

### Resources used by the Modal app
- GPU: H100
- Timeout: 4 hours for `run_all`, 2 hours for individual functions
- Max concurrent GPU containers: 5
- Hugging Face cache persisted via a Modal Volume
- Results copied to a persistent `benchmarks` Volume at `/benchmarks/<model>/scores/`

### Run all three benchmarks (alignment, diversity, text)
Call the single cloud function `run_all` to run all benchmarks for a single model:

```bash
modal run modal/apps/OneIG-Benchmark/start_benchmark_modal.py::run_all \
  --model-name gpt-4o \
  --image-grid '[2,2]' \
```

To run benchmarks for multiple models, call the function separately for each model:
```bash
# For gpt-4o
modal run modal/apps/OneIG-Benchmark/start_benchmark_modal.py::run_all \
  --model-name gpt-4o \
  --image-grid '[2,2]' \

# For imagen4  
modal run modal/apps/OneIG-Benchmark/start_benchmark_modal.py::run_all \
  --model-name imagen4 \
  --image-grid '[2,2]' \
```

Notes:
- Alignment runs with class items: `human object` (no `anime`).
- Diversity runs with class items: `human object text`.
- Text benchmark reads from `/images/text/<model>/<id>.webp`.

### Run benchmarks individually
You can invoke each cloud function directly if you prefer to run them one at a time, using the same shared model/grid configuration for all tests.

#### Alignment
```bash
modal run modal/apps/OneIG-Benchmark/start_benchmark_modal.py::run_alignment \
  --model-name gpt-4o \
  --image-grid '[2,2]'
```
Behavior: executes `scripts.alignment.alignment_score` with `--class_items human object`.

#### Diversity
```bash
modal run modal/apps/OneIG-Benchmark/start_benchmark_modal.py::run_diversity \
  --model-name gpt-4o \
  --image-grid '[2]'
```
Behavior: executes `scripts.diversity.diversity_score` with `--class_items human object text`. Note: diversity defaults to `[2]` grid.

#### Text
```bash
modal run modal/apps/OneIG-Benchmark/start_benchmark_modal.py::run_text \
  --model-name gpt-4o \
  --image-grid '[2,2]'
```
Behavior: executes `scripts.text.text_score` against the `/images/text` subdirectory.

### Outputs
- Each run writes CSVs under the repo at `/oneig/results/` and copies them to the persistent volume paths:
  - Alignment: `/benchmarks/<model>/scores/alignment_score_EN_*.csv`
  - Diversity: `/benchmarks/<model>/scores/diversity_score_EN_*.csv`
  - Text: `/benchmarks/<model>/scores/text_score_EN_*.csv`

### Tips
- Adjust `--mode ZH` only if your images and prompts are from the ZH dataset.
- `--image-grid` should match how many tiles you generated per prompt: `1` for single, `2` for a 2x2 grid, etc.
- Each function processes a single model at a time. To benchmark multiple models, run the functions separately for each model.
- The `run_all` function has a 4-hour timeout to accommodate running all three benchmarks sequentially.
- Diversity benchmark defaults to `[2]` grid, while alignment and text default to `[2,2]`.
