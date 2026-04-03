#!/bin/bash

# Usage: ./run_experiments.sh ditto   OR   ./run_experiments.sh fedavg
METHOD=$1  # first argument: "ditto" or "fedavg"

if [[ "$METHOD" != "ditto" && "$METHOD" != "fedavg" ]]; then
    echo "Error: You must specify method as 'ditto' or 'fedavg'."
    exit 1
fi

# Set variables
DATASET="mnist"
GPUID=0
DATA_TYPES=("iid" "niid")  # add more data types if needed
SEEDS=(46)        # add your seeds here

for DATA_TYPE in "${DATA_TYPES[@]}"; do
    echo "=== Running experiments for data_type: $DATA_TYPE ==="
    
    for SEED in "${SEEDS[@]}"; do
        echo "--- Seed $SEED ---"

        if [[ "$METHOD" == "ditto" ]]; then
            echo "Running ditto_vanilla.py..."
            python ditto_vanilla.py --dataset $DATASET --gpuid $GPUID --seed $SEED --data_type $DATA_TYPE

            echo "Running ditto_unidp.py..."
            python ditto_unidp.py --dataset $DATASET --gpuid $GPUID --seed $SEED --data_type $DATA_TYPE

            echo "Running ditto_rpdp.py..."
            python ditto_rpdp.py --dataset $DATASET --gpuid $GPUID --seed $SEED --data_type $DATA_TYPE
        else
            echo "Running fedavg_vanilla.py..."
            python fedavg_vanilla.py --dataset $DATASET --gpuid $GPUID --seed $SEED --data_type $DATA_TYPE

            echo "Running fedavg_unidp.py..."
            python fedavg_unidp.py --dataset $DATASET --gpuid $GPUID --seed $SEED --data_type $DATA_TYPE

            echo "Running fedavg_rpdp.py..."
            python fedavg_rpdp.py --dataset $DATASET --gpuid $GPUID --seed $SEED --data_type $DATA_TYPE
        fi

        echo "--- Finished seed $SEED ---"
    done

    echo "=== Finished all seeds for data_type: $DATA_TYPE ==="
done

echo "All experiments completed!"