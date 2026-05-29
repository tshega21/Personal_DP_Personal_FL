

# dataset flags: {"heart_disease","fashion_mnist", "mnist", "cifar10"}
# "iid_10" "niid_10_5" "niid_10_2" "niid_dir_1" "niid_dir_5"

#!/bin/bash

# chmod +x the_figure_3_lambda_sweep_experiments_DP.sh
# ./the_figure_3_lambda_sweep_experiments_DP.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="mnist"
GPU=0
#"iid_10" "niid_10_5" "niid_10_2" "niid_dir_1" "niid_dir_5"
DATA_TYPES=("iid_10")
REG_PARAMS=(0 0.001 0.01 0.1 1.0 10)
 #0.1 0.5 1 8
EPSILONS=(4)
NUM_STEPS=15
SEEDS=(42)

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

for DATA_TYPE in "${DATA_TYPES[@]}"; do

    LOG_FILE="$LOG_DIR/the_figure_3_lambda_sweep_DP_15_${DATASET}_${DATA_TYPE}_$(date +%Y%m%d_%H%M%S).log"

    echo "Logging to $LOG_FILE"
    echo "PARAMS: dataset=$DATASET, gpu=$GPU, data_type=$DATA_TYPE, num_steps=$NUM_STEPS" | tee -a "$LOG_FILE"

    for EPS in "${EPSILONS[@]}"; do
        echo "=== Running epsilon=$EPS ===" | tee -a "$LOG_FILE"

        for REG in "${REG_PARAMS[@]}"; do
            echo "=== Running reg_param=$REG ===" | tee -a "$LOG_FILE"

            for SEED in "${SEEDS[@]}"; do
                echo "Running seed $SEED" | tee -a "$LOG_FILE"
                echo "Running ditto_unidp.py..."


                python ditto_unidp.py \
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

echo "All runs complete."