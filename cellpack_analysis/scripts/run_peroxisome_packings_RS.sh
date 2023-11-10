#!/bin/bash
python /allen/aics/animated-cell/Saurabh/cellpack-analysis/cellpack_analysis/scripts/run_packings_for_structure.py \
--structure_name "peroxisome" \
--structure_id "SLC25A17" \
--out_folder "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/packing_outputs/RS" \
--config_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/configs/peroxisome_packing_config_mean.json" \
--generated_recipe_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/generated_recipes/peroxisome_RS" \
--datadir "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/" \
--recipe_template_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/templates/peroxisome_mean_template.json" \
--num_processes 32 \
--run_packings \
--skip_completed \
--use_cellid_as_seed \
--generate_recipes \
# --dry_run \
# --num_packings 4 \