#!/bin/bash
python /allen/aics/animated-cell/Saurabh/cellpack-analysis/cellpack_analysis/scripts/run_packings_for_structure.py \
--structure_name "endosome" \
--structure_id "RAB5A" \
--out_folder "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/packing_outputs/RVC" \
--config_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/configs/endosome_packing_config_RVC.json" \
--generated_recipe_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/generated_recipes/endosome_RVC" \
--datadir "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/" \
--recipe_template_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/templates/endosome_variable_template.json" \
--num_processes 16 \
--run_packings \
--skip_completed \
--generate_recipes \
--use_mean_cell \
# --dry_run \
# --use_cellid_as_seed \
# --num_packings 4 \