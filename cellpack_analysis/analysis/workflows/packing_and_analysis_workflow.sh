#! /bin/bash
PACKING_CONFIG_NAME="$1"
ANALYSIS_CONFIG_NAME="$2"
python cellpack_analysis/packing/run_packing_workflow.py \
    -c "cellpack_analysis/packing/configs/$1.json"

python cellpack_analysis/analysis/workflows/run_analysis_workflow.py \
    -c "cellpack_analysis/analysis/workflows/configs/$2.json"