#!/bin/bash

# Usage: ./run_experiment_hd.sh ditto   OR   ./run_experiments_hd.sh fedavg
METHOD=$1  # first argument: "ditto" or "fedavg"

if [[ "$METHOD" != "ditto" && "$METHOD" != "fedavg" ]]; then
    echo "Error: You must specify method as 'ditto' or 'fedavg'."
    exit 1
fi


# dataset flags: {"heart_disease", "mnist", "cifar10", "snli"}


# Set variables
DATASET="heart_disease"
GPUID=0
DATA_TYPES=("iid")  
VERSIONS=("ditto_vanilla" "ditto_rpdp" "dp_ditto_rpdp")
SEEDS=(42 43 44 45 46)        # add seeds here
REG_PARAMS=(0.001 0.05 0.01 0.1 1.0 2.5 5 7.5 10)  # lambda for ditto #0 0.001 0.05 0.01 0.1 1.0 2.5 5 7.5 10
NUM_PERSONAL_STEPS=(10) # number of personalization steps for ditto # 1 2 5 7 10 12 15
EPSILONS=(0.1 0.5 1 2.5 5 7.5 10)   # values of epsilon for uniform record dp

for DATA_TYPE in "${DATA_TYPES[@]}"; do
    echo "=== Running experiments for data_type: $DATA_TYPE ==="
    
    for SEED in "${SEEDS[@]}"; do
        echo "--- Seed $SEED ---"

        if [[ "$METHOD" == "ditto" ]]; then

            for NUM_PERSONAL_STEP in "${NUM_PERSONAL_STEPS[@]}"; do 
                echo "--- Number of Personal Steps $NUM_PERSONAL_STEP ---"

            
                for REG_PARAM in "${REG_PARAMS[@]}"; do 
                echo "--- Lambda $REG_PARAM ---"
                
                    for VERSION in "${VERSIONS[@]}"; do
                        echo "Running $VERSION.py..."

                        python $VERSION.py --dataset $DATASET --gpuid $GPUID --seed $SEED --data_type $DATA_TYPE --reg_param $REG_PARAM --num_personal_steps $NUM_PERSONAL_STEP &


                    #echo "Running ditto_rpdp.py..."
                    #python ditto_rpdp.py --dataset $DATASET --gpuid $GPUID --seed $SEED --data_type $DATA_TYPE --reg_param $REG_PARAM --num_personal_steps $NUM_PERSONAL_STEP &



                    #echo "Running dp_ditto_rpdp.py..."
                    #python dp_ditto_rpdp.py --dataset $DATASET --gpuid $GPUID --seed $SEED --data_type $DATA_TYPE --reg_param $REG_PARAM --num_personal_steps $NUM_PERSONAL_STEP &

                    #for EPSILON in "${EPSILONS[@]}"; do 
                    #    echo "---EPSILON $EPSILON ---"

                    #    echo "Running ditto_unidp.py..."
                    #    python ditto_unidp.py --dataset $DATASET --gpuid $GPUID --seed $SEED --data_type $DATA_TYPE --epsilon $EPSILON --reg_param $REG_PARAM --num_personal_steps $NUM_PERSONAL_STEP


                    #    echo "Running dp_ditto_unidp.py..."
                    #    python dp_ditto_unidp.py --dataset $DATASET --gpuid $GPUID --seed $SEED --data_type $DATA_TYPE
                   # done
                   done
                   wait
                done
            done
        else
            echo "Running fedavg_vanilla.py..."
            python fedavg_vanilla.py --dataset $DATASET --gpuid $GPUID --data_type $DATA_TYPE --seed $SEED &

            echo "Running fedavg_rpdp.py..."
            python fedavg_rpdp.py --dataset $DATASET --gpuid $GPUID --data_type $DATA_TYPE --seed $SEED &

            wait
        
            for EPSILON in "${EPSILONS[@]}"; do
                echo "---EPSILON $EPSILON ---"
                echo "Running fedavg_unidp.py..."
                python fedavg_unidp.py --dataset $DATASET --gpuid $GPUID --data_type $DATA_TYPE --epsilon $EPSILON --seed $SEED 
            done
        fi

        echo "--- Finished seed $SEED ---"
    done

    echo "=== Finished all seeds for data_type: $DATA_TYPE ==="
done

echo "All experiments completed!"