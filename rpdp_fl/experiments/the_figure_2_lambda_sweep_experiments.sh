#!/bin/bash

# chmod +x the_figure2_lambda_sweep_experiments.sh
# ./the_figure_2_lambda_sweep_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# dataset flags: {"heart_disease","fashion_mnist", "mnist", "cifar10"}

# mnist: set fedavg: learning_rate = 0.001 for non iid in .yaml config to avoid erratic convergence 
# cifar10: learning_rate = 0.001 in .yaml config 
# fashion_mnist: learning_rate = 0.001 for both iid and noniid  in .yaml config 

# "iid_10" "niid_10_5" "niid_10_2" "niid_dir_1" "niid_dir_5" 
DATASET="fashion_mnist"
GPU=0
DATA_TYPES=("niid_10_5" "niid_10_2" "niid_dir_1" "niid_dir_5" )  # Add all data types you want to iterate over
REG_PARAMS=(0 0.001 0.01 0.1 1.0  5 10 15 20)        # Add all regularization parameters you want to test
NUM_STEPS=30
SEEDS=(42)

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

for DATA_TYPE in "${DATA_TYPES[@]}"; do
    LOG_FILE="$LOG_DIR/the_figure_2_lambda_sweep_30_${DATASET}__${DATA_TYPE}_$(date +%Y%m%d_%H%M%S).log"
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
echo "All runs complete." | tee -a "$LOG_FILE"