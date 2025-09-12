#!/bin/bash
python /allen/aics/animated-cell/Saurabh/cellpack-analysis/cellpack_analysis/scripts/run_packings_for_structure.py \
--structure_name "endosome" \
--structure_id "RAB5A" \
--out_folder "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/packing_outputs/20231103_replicates_variable_count_size" \
--config_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/configs/endosome_packing_config_replicates.json" \
--generated_recipe_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/generated_recipes/20231103_endosome_replicates_variable_count_size" \
--datadir "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/" \
--recipe_template_path "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/templates/endosome_variable_template.json" \
--num_processes 32 \
--run_packings \
--generate_recipes \
--num_packings 8 \
--use_cell_id_as_seed \
--skip_completed \
# --dry_run \