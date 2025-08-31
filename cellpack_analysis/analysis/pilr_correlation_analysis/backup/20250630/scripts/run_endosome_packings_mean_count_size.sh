#!/bin/bash
python /allen/aics/animated-cell/Saurabh/cellpack-analysis/cellpack_analysis/scripts/run_packings_for_structure.py \
--structure_name "endosome" \
--structure_id "RAB5A" \
--out_folder "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/packing_outputs/mean_count_and_size" \
--config_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/configs/endosome_packing_config_mean.json" \
--generated_recipe_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/generated_recipes/endosome_mean_count_size" \
--datadir "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/" \
--recipe_template_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/templates/endosome_mean_template.json" \
--num_processes 16 \
--run_packings \
--use_cell_id_as_seed \
--skip_completed \
# --dry_run \
# --generate_recipes \
# --num_packings 4 \