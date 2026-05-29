#!/bin/bash

# chmod +x the_figure_2_personal.sh
# ./the_figure_2_personal.sh 

#IID 


#!/bin/bash

# chmod +x the_figure_2_personal.sh
# ./the_figure_2_personal.sh 

#IID 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#"fashion_mnist" "mnist" 
# datasets to iterate over
DATASETS=("mnist")
GPU=0
DATA_TYPES=("niid_dir_1")  # Add all data types you want to iterate over
REG_PARAMS=(10)
NUM_STEPS=15
SEEDS=(42 43 44 45 46)

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

for DATASET in "${DATASETS[@]}"; do
    for DATA_TYPE in "${DATA_TYPES[@]}"; do
        LOG_FILE="$LOG_DIR/the_figure_2_personal_${DATASET}__${DATA_TYPE}_$(date +%Y%m%d_%H%M%S).log"
        echo "Logging to $LOG_FILE"
        echo "PARAMS: dataset=$DATASET, gpu=$GPU, data_type=$DATA_TYPE, num_steps=$NUM_STEPS" | tee -a "$LOG_FILE"

        for REG in "${REG_PARAMS[@]}"; do
            echo "=== Running reg_param=$REG ===" | tee -a "$LOG_FILE"

            for SEED in "${SEEDS[@]}"; do
                echo "Running seed $SEED" | tee -a "$LOG_FILE"

                python ditto_vanilla.py \
                    --dataset "$DATASET" \
                    --gpuid "$GPU" \
                    --seed "$SEED" \
                    --data_type "$DATA_TYPE" \
                    --reg_param "$REG" \
                    --num_personal_steps "$NUM_STEPS" \
                    2>&1 | tee -a "$LOG_FILE"

                echo "Finished seed $SEED" | tee -a "$LOG_FILE"
                echo "----------------------------------------" | tee -a "$LOG_FILE"
            done
        done
    done
done
echo "All runs complete." | tee -a "$LOG_FILE"
