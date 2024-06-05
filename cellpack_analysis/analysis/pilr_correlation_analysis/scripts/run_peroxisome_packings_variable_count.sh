#!/bin/bash
python /allen/aics/animated-cell/Saurabh/cellpack-analysis/cellpack_analysis/scripts/run_packings_for_structure.py \
--structure_name "peroxisome" \
--structure_id "SLC25A17" \
--out_folder "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/packing_outputs/variable_count" \
--config_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/configs/peroxisome_packing_config_variable_count.json" \
--generated_recipe_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/generated_recipes/peroxisome_variable_count" \
--datadir "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/" \
--recipe_template_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/templates/peroxisome_variable_count_template.json" \
--num_processes 1 \
--run_packings \
--generate_recipes \
--use_mean_cell \
--skip_completed \
--use_cellid_as_seed \
# --dry_run \
# --num_packings 4 \