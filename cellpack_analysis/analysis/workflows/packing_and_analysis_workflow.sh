#! /bin/bash
CONFIG_NAME="$1"
echo "python cellpack_analysis/packing/run_packing_workflow.py \
-c cellpack_analysis/packing/configs/$CONFIG_NAME.json" && \
python cellpack_analysis/packing/run_packing_workflow.py \
    -c "cellpack_analysis/packing/configs/$CONFIG_NAME.json" && \
echo "python cellpack_analysis/analysis/workflows/run_analysis_workflow.py \
-c cellpack_analysis/analysis/workflows/configs/analysis_config_$CONFIG_NAME.json" && \
python cellpack_analysis/analysis/workflows/run_analysis_workflow.py \
    -c "cellpack_analysis/analysis/workflows/configs/analysis_config_$CONFIG_NAME.json"