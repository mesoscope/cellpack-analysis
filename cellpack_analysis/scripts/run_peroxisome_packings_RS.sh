#!/bin/bash
python /allen/aics/animated-cell/Saurabh/cellpack-analysis/cellpack_analysis/scripts/run_packings_for_structure.py \
--structure_name "peroxisome" \
--structure_id "SLC25A17" \
--out_folder "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/packing_outputs/stochastic_variation_analysis/shape" \
--config_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/configs/peroxisome_packing_config_shape_noimage.json" \
--generated_recipe_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/generated_recipes/stochastic_variation_analysis/peroxisome_shape" \
--datadir "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/" \
--recipe_template_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/templates/peroxisome_mean_template.json" \
--num_processes 32 \
--run_packings \
--skip_completed \
--use_cellid_as_seed \
--generate_recipes \
# --dry_run \
# --num_packings 4 \