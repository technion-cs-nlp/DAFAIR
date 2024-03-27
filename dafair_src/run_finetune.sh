#!/bin/bash

#SBATCH --job-name=main_r
#SBATCH --output=outputs_reports/output_%j.txt
#SBATCH --partition=nlp
#SBATCH --account=nlp 
#SBATCH --exclude=nlp-ada-1
#SBATCH --exclude=nlp-ada-2
#SBATCH --exclude=nlp-a40-1
# Define the experiment parameters
models=("bert")
run_ids=(0 1 2 3 4)
emb_modes=(-1)
set_embs=(1)
lam_ends=(0)
iters=10000

# Define a function to run the experiment
run_experiment() {
    model=$1
    run_id=$2
    emb_modes=$3
    set_embs=$4
    lam_end=$5
    adv=$6
    
    # Define the command to run the experiment
    command="python3 -u finetune_dafair.py --model ${model} --run_id ${run_id} --emb_mode ${emb_modes} --set_emb ${set_embs} --lam_end ${lam_end} --labaled_size 1000 --iters ${iters} --save 1 --learned_loss 1 --proj_name firstres"
    
    echo "${command}"
    
    # Run the command using sbatch
    sbatch <<< "#!/bin/bash
#SBATCH --job-name=my_experiment_${model}_${run_id}_${lam_end}_${emb_modes}
#SBATCH --output=outputs_reports/dd_output_${model}_${run_id}_${lam_end}_${emb_modes}_%j.txt
#SBATCH --partition=nlp
#SBATCH --account=nlp
#SBATCH --gres=gpu:1
#SBATCH --exclude=nlp-ada-1
#SBATCH --exclude=nlp-ada-2
#SBATCH --exclude=nlp-a40-1
${command}"
}


# Loop over the parameter combinations and run the experiments
for model in "${models[@]}"
do
    for run_id in "${run_ids[@]}"
    do
        for lam_end in "${lam_ends[@]}"
        do
            for emb_mode in "${emb_modes[@]}"
            do
                for set_emb in "${set_embs[@]}"
                do
                    # Run the experiment in the background
                    run_experiment "${model}" "${run_id}" "${emb_mode}" "${set_emb}" "${lam_end}" "${adv}" &
                done
            done
        done
    done
    sleep 30m
done
# Wait for all background processes to finish
wait
