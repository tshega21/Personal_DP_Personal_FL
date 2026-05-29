#!/bin/bash

#chmod +x the_figure_1_total_experiments.sh
#./the_figure_1_total_experiments.sh


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# ========================
# Configuration
# ========================
DATASET="cifar10"
GPUID=0

# dataset flags: {"heart_disease","fashion_mnist", "mnist", "cifar10"}
# mnist: set fedavg: learning_rate = 0.001 in .yaml config to avoid erratic convergence 
# cifar10: learning_rate = 0.001 in .yaml config 
# fashion_mnist: learning_rate = 0.001 in .yaml config 
#"iid_10"
# dataset type flags: {"niid_10_5" "niid_10_2" "niid_dir_1" "niid_dir_5"}


DATA_TYPES=( "niid_dir_1" "niid_dir_5")
SEEDS=(42 43 44 45 46)
REG_PARAM=(0)  #local training

EPSILONS=( 0.1 0.5 1 4 8 16)   # values of epsilon for uniform record dp

NUM_PERSONAL_STEPS=(15)

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# ========================
# Run Experiments
# ========================
for DATA_TYPE in "${DATA_TYPES[@]}"; do
    LOG_FILE="$LOG_DIR/the_figure1_total_15_30_${DATASET}_${DATA_TYPE}_$(date +%Y%m%d_%H%M%S).log"

    echo "Logging to $LOG_FILE"
    echo "=== Data type: $DATA_TYPE ===" | tee -a "$LOG_FILE"

    for SEED in "${SEEDS[@]}"; do
        echo "--- Seed: $SEED ---" | tee -a "$LOG_FILE"

        for NUM_PERSONAL_STEP in "${NUM_PERSONAL_STEPS[@]}"; do
            echo "--- Personal steps: $NUM_PERSONAL_STEP ---" | tee -a "$LOG_FILE"

            for EPSILON in "${EPSILONS[@]}"; do 
                echo "---EPSILON $EPSILON ---" | tee -a "$LOG_FILE"

                echo "Running dp_ditto_unidp.py..."
                python dp_ditto_unidp.py --dataset $DATASET \
                                         --gpuid $GPUID \
                                         --seed $SEED \
                                         --data_type $DATA_TYPE \
                                         --epsilon $EPSILON \
                                         --reg_param $REG_PARAM \
                                         --num_personal_steps $NUM_PERSONAL_STEP \
                                           2>&1 | tee -a "$LOG_FILE"

                wait
            done
        done

        echo "--- Finished seed: $SEED ---" | tee -a "$LOG_FILE"
    done

    echo "=== Finished data_type: $DATA_TYPE ===" | tee -a "$LOG_FILE"
done

echo "All experiments completed!" | tee -a "$LOG_FILE"