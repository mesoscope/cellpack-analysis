#!/bin/bash

STRUCTURE_ID="$1"
STRUCTURE_NAME="$2"
FOLDER_ID="$3"
CHANNEL_NAMES="$4"

python "$cellpack_analysis/cellpack_analysis/analysis/pilr_correlation_analysis/workflows/calculate_PILR_from_images.py" \
  --raw_image_path "$cellpack_analysis/data/structure_data/$STRUCTURE_ID/sample_8d/segmented" \
  --simulated_image_path "$cellpack_analysis/data/packing_outputs/8d_sphere_data/$FOLDER_ID/{rule}/$STRUCTURE_NAME/spheresSST/figures/" \
  --save_dir "$cellpack_analysis/data/PILR/$STRUCTURE_NAME/$FOLDER_ID" \
  --raw_image_channel "$STRUCTURE_ID" \
  --channel_names $CHANNEL_NAMES \
  --num_cores 64

