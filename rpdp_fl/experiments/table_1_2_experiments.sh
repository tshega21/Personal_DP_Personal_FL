#!/bin/bash

#chmod +x table_1_2_experiments.sh
#./table_1_2_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# dataset flags: {"heart_disease","fashion_mnist", "mnist", "cifar10"}

DATASET="cifar10"
GPU=0
DATA_TYPE="iid_10"
# local training when reg param = 0
REG=0.0
NUM_STEPS=15

SEEDS=(42 43 44 45 46)

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/table_1_2_ditto_${DATASET}__${DATA_TYPE}_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to $LOG_FILE"
echo "PARAMS: dataset=$DATASET, gpu=$GPU, data_type=$DATA_TYPE, reg=$REG, steps=$NUM_STEPS" | tee -a "$LOG_FILE"

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