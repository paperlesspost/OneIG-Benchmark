#!/bin/bash

# start_time
start_time=$(date +%s)

# mode (EN/ZH)
MODE=EN

# image_root_dir
IMAGE_DIR=""

# model list
MODEL_NAMES=("gpt-4o" "imagen4")
# model_names=("gpt-4o" "imagen4")

# image grid
IMAGE_GRID=(2 2)

# Reasoning Score

echo "It's reasoning time."

pip install transformers==4.50.0

python -m scripts.reasoning.reasoning_score \
  --mode "$MODE" \
  --image_dirname "${IMAGE_DIR}/reasoning" \
  --model_names "${MODEL_NAMES[@]}" \
  --image_grid "${IMAGE_GRID[@]}" \

# end_time
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "✅ All evaluations finished in $duration seconds."