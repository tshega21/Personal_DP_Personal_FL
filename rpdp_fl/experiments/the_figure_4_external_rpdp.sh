#!/bin/bash

#chmod +x the_figure_4_external_rpdp.sh
#./the_figure_4_external_rpdp.sh

#ESSENTIALLY DOES A LAMBDA SWEEP, INCLUDES LOCAL TRAINING 


# dataset flags: {"heart_disease","fashion_mnist", "mnist", "cifar10"}

# mnist: set fedavg: learning_rate = 0.001 for non iid in .yaml config to avoid erratic convergence 
# cifar10: learning_rate = 0.001 in .yaml config 
# fashion_mnist: learning_rate = 0.001 for both iid and noniid  in .yaml config 

# "iid_10" "niid_10_5" "niid_10_2" "niid_dir_1" "niid_dir_5" 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="cifar10"
GPU=0

DATA_TYPES=("iid_10" "niid_10_5" "niid_10_2" "niid_dir_1" "niid_dir_5")
REG_PARAMS=(0 0.001 0.01 0.1 1.0 10 20)
EPS=5
NUM_STEPS_LIST=(15 30)

SEEDS=(42)

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

for DATA_TYPE in "${DATA_TYPES[@]}"; do
    LOG_FILE="$LOG_DIR/the_figure_4_external_rpdp_${DATASET}__${DATA_TYPE}_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to $LOG_FILE"
    echo "PARAMS: dataset=$DATASET, gpu=$GPU, data_type=$DATA_TYPE" | tee -a "$LOG_FILE"

    for REG in "${REG_PARAMS[@]}"; do
        for NUM_STEPS in "${NUM_STEPS_LIST[@]}"; do
            echo "=== Running reg_param=$REG, num_steps=$NUM_STEPS ===" | tee -a "$LOG_FILE"

            for SEED in "${SEEDS[@]}"; do
                echo "Running seed $SEED" | tee -a "$LOG_FILE"

                python ditto_rpdp.py \
                    --dataset "$DATASET" \
                    --gpuid "$GPU" \
                    --seed "$SEED" \
                    --data_type "$DATA_TYPE" \
                    --epsilon "$EPS" \
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