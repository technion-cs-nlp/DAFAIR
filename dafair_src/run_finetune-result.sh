#!/bin/bash

#SBATCH --job-name=main_r
#SBATCH --output=outputs_reports/output_%j.txt
#SBATCH --partition=nlp
#SBATCH --account=nlp
# Define the experiment parameters
models=("bert")

# Define a function to run the experiment
run_experiment-moji() {
    model=$1
    run_id=$2
    emb_mode=$3
    set_emb=$4
    lam_end=$5
    labeled_size=$6
    # Define the command to run the experiment
    command="python3 -u finetune_dafair.py --model ${model} --run_id ${run_id} --dataset moji --emb_mode ${emb_mode} --set_emb ${set_emb} --lam_end ${lam_end} --labaled_size ${labeled_size} --save 1 --learned_loss 1 --proj_name Moji"
    
    echo "${command}"
    
    # Run the command using sbatch
    sbatch <<< "#!/bin/bash
#SBATCH --job-name=my_experiment_${model}_${run_id}_${lam_end}_${emb_mode}
#SBATCH --output=outputs_reports/dd_output_${model}_${run_id}_${lam_end}_${emb_mode}_%j.txt
#SBATCH --partition=nlp
#SBATCH --account=nlp
#SBATCH --gres=gpu:1
${command}"
}

run_experiment-bios() {
    model=$1
    run_id=$2
    emb_mode=$3
    set_emb=$4
    lam_end=$5
    labeled_size=$6
    # Define the command to run the experiment
    command="python3 -u finetune_dafair.py --model ${model} --run_id ${run_id} --dataset bios --emb_mode ${emb_mode} --set_emb ${set_emb} --lam_end ${lam_end} --labaled_size ${labeled_size} --save 1 --learned_loss 1 --proj_name BIOS"
    
    echo "${command}"
    
    # Run the command using sbatch
    sbatch <<< "#!/bin/bash
#SBATCH --job-name=my_experiment_${model}_${run_id}_${lam_end}_${emb_mode}
#SBATCH --output=outputs_reports/dd_output_${model}_${run_id}_${lam_end}_${emb_mode}_%j.txt
#SBATCH --partition=nlp
#SBATCH --account=nlp
#SBATCH --gres=gpu:1
${command}"
}

run_ids=()
#labels_sizes=(10000 100000 255710)
labels_sizes=(100)
#$lab=100
# Run each case with 5 different run_ids
for run_id in "${run_ids[@]}"; do
  for lab in "${labels_sizes[@]}"; do

    #bios
    run_experiment-bios "deberta" "$run_id" 0 4 0.5 "$lab"
    
    #moji
    run_experiment-moji "deberta" "$run_id" 0 4 0.01 "$lab"
 
  done
done
# Wait for all background processes to finish
wait


