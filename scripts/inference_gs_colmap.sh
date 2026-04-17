#!/usr/bin/env bash
# Run DepthSplat Gaussian-Splatting inference on a COLMAP scene and export .ply.
#
#   1. Convert COLMAP -> DepthSplat .torch chunk.
#   2. Download the multi-view GS checkpoint if missing.
#   3. Run `src.main mode=test` with test.save_gaussian=true.
#
# Output: outputs/${OUTPUT_NAME}/gaussians/${SCENE_KEY}.ply

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-/home/mas/proj/study/strayscanner_converter/output/sensyn_building}"
SCENE_KEY="${SCENE_KEY:-sensyn_building}"
DATASET_DIR="${DATASET_DIR:-datasets/custom_colmap}"
OUTPUT_NAME="${OUTPUT_NAME:-depthsplat-gs-${SCENE_KEY}}"

CHECKPOINT="pretrained/depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth"
CHECKPOINT_URL="https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth"

# 4 evenly spaced context views over 825 frames and one arbitrary target.
# (The checkpoint supports 4-10 views; 4 keeps cost-volume memory low enough
# for ~12 GB GPUs. Raise NUM_CONTEXT / CONTEXT_VIEWS on bigger hardware.)
CONTEXT_VIEWS="${CONTEXT_VIEWS:-[0,275,549,824]}"
TARGET_VIEWS="${TARGET_VIEWS:-[412]}"
NUM_CONTEXT="${NUM_CONTEXT:-4}"
IMAGE_SHAPE="${IMAGE_SHAPE:-[448,768]}"

# 1. Convert COLMAP -> .torch if not already present.
if [[ ! -f "${DATASET_DIR}/test/000000.torch" ]]; then
  echo "=== Converting COLMAP data ==="
  uv run python scripts/convert_colmap_to_depthsplat.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${DATASET_DIR}" \
    --scene_key "${SCENE_KEY}"
else
  echo "=== Reusing existing chunk: ${DATASET_DIR}/test/000000.torch ==="
fi

# 2. Download checkpoint if missing.
mkdir -p pretrained
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "=== Downloading checkpoint ==="
  wget -P pretrained "${CHECKPOINT_URL}"
fi

# 3. Run GS inference.
echo "=== Running GS inference ==="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python -m src.main +experiment=re10k \
  mode=test \
  dataset.roots=["${DATASET_DIR}"] \
  dataset/view_sampler=arbitrary \
  dataset.view_sampler.num_context_views="${NUM_CONTEXT}" \
  dataset.view_sampler.num_target_views=1 \
  dataset.view_sampler.context_views="${CONTEXT_VIEWS}" \
  +dataset.view_sampler.target_views="${TARGET_VIEWS}" \
  dataset.image_shape="${IMAGE_SHAPE}" \
  dataset.skip_bad_shape=false \
  model.encoder.upsample_factor=8 \
  model.encoder.lowest_feature_resolution=8 \
  model.encoder.gaussian_adapter.gaussian_scale_max=0.1 \
  checkpointing.pretrained_model="${CHECKPOINT}" \
  test.compute_scores=false \
  test.save_gaussian=true \
  output_dir="outputs/${OUTPUT_NAME}"

echo
echo "Done. PLY: outputs/${OUTPUT_NAME}/gaussians/${SCENE_KEY}.ply"
