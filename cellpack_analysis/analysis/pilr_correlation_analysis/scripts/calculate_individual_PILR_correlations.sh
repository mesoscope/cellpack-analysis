#!/bin/bash
STRUCTURE_ID=$1
STRUCTURE_NAME=$2
FOLDER_ID=$3
python $home/cellpack-analysis/cellpack_analysis/scripts/individual_PILR_correlation.py \
--structure_id $STRUCTURE_ID \
--structure_name $STRUCTURE_NAME \
--base_folder $home/cellpack-analysis/results/$STRUCTURE_ID/$FOLDER_ID/ \
--save_individual_images \
# --get_correlations \