#!/bin/bash

#chmod +x table_2_3_experiments.sh
#./table_2_3_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


DATASET="mnist"
GPU=0

# dataset flags: {"heart_disease","fashion_mnist", "mnist", "cifar10"}
# mnist: set fedavg: learning_rate = 0.001 in .yaml config to avoid erratic convergence 
# cifar10: learning_rate = 0.001 in .yaml config 
# fashion_mnist: learning_rate = 0.001 in .yaml config 

# dataset type flags: {"niid_10_5","niid_10_2", "niid_dir_1", "niid_dir_2"}

DATA_TYPE="iid_10"
# local training when reg param = 0
REG=0.0
NUM_STEPS=15

SEEDS=(42 43 44 45 46)

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/table_2_3_ditto_${DATASET}_${DATA_TYPE}_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to $LOG_FILE"

for SEED in "${SEEDS[@]}"
do
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

echo "All runs complete." | tee -a "$LOG_FILE"