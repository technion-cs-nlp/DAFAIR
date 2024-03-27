#!/bin/bash

#SBATCH --job-name=main_r
#SBATCH --output=outputs_reports/output_%j.txt
#SBATCH --partition=nlp
#SBATCH --account=nlp
# Define the experiment parameters

# Define a function to run the experiment
run_experiment-moji() {
    model=$1
    run_id=$2
    emb_mode=$3
    set_emb=$4
    lam_end=$5
    labeled_size=$6
    # Define the command to run the experiment
    command="python3 -u finetune_dafair.py --model ${model} --run_id ${run_id} --dataset moji --emb_mode ${emb_mode} --set_emb ${set_emb} --lam_end ${lam_end} --labaled_size ${labeled_size} --save 0 --learned_loss 1 --proj_name tuning"
    
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
    command="python3 -u finetune_dafair.py --model ${model} --run_id ${run_id} --dataset bios --emb_mode ${emb_mode} --set_emb ${set_emb} --lam_end ${lam_end} --labaled_size ${labeled_size} --save 0 --learned_loss 1 --proj_name tuning"
    
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


model1="bert"
model2="deberta-base"
model3="roberta"
run_ids=()
emb_set_values_pd=(4)
lambda_values_pd=(1 10 25 50 75 100)

emb_set_values_dd=(4)
lambda_values_dd=(1 10 25 50 75 100)


lab=100
lambda_values_moji=(0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10)
for run_id in "${run_ids[@]}"; do

  sleep 1h
  run_experiment-moji "$model1" "$run_id" -1 4 0 "$lab"

  for l in "${lambda_values_moji[@]}"; do
    #moji
    run_experiment-moji "$model1" "$run_id" 0 4 "$l" "$lab"
    #bios
    run_experiment-bios "$model1" "$run_id" 0 4 "$l" "$lab"

    sleep 1h
    
  done
  sleep 1h
done



# Wait for all background processes to finish
wait
