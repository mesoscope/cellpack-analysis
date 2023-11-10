#!/bin/bash
STRUCTURE_ID=$1
STRUCTURE_NAME=$2
FOLDER_ID=$3
python $home/cellpack-analysis/cellpack_analysis/scripts/cellPACK_PILR.py \
--raw_image_path $home/cellpack-analysis/data/structure_data/$STRUCTURE_ID/sample_8d/raw_imgs_for_PILR \
--raw_image_channel $STRUCTURE_ID \
--simulated_image_path $home/cellpack-analysis/data/packing_outputs/$FOLDER_ID/$STRUCTURE_NAME/spheresSST/figures/ \
--save_dir $home/cellpack-analysis/results/$STRUCTURE_ID/$FOLDER_ID \
--num_cores 32 \
