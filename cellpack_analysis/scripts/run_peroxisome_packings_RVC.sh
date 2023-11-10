#!/bin/bash
python /allen/aics/animated-cell/Saurabh/cellpack-analysis/cellpack_analysis/scripts/run_packings_for_structure.py \
--structure_name "peroxisome" \
--structure_id "SLC25A17" \
--out_folder "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/packing_outputs/RVC" \
--config_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/configs/peroxisome_packing_config_RVC.json" \
--generated_recipe_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/generated_recipes/peroxisome_RVC" \
--datadir "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/" \
--recipe_template_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/templates/peroxisome_variable_template.json" \
--num_processes 16 \
--run_packings \
--skip_completed \
--generate_recipes \
--use_mean_cell \
# --dry_run \
# --use_cellid_as_seed \
# --num_packings 4 \