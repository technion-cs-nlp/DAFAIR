#!/bin/bash

#SBATCH --job-name=main_r
#SBATCH --output=outputs_reports/eval_%j.txt
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
    dataset=$7
    # Define the command to run the experiment
    command="python3 -u evaluate_dafair.py --model ${model} --run_id ${run_id} --dataset ${dataset} --emb_mode ${emb_mode} --set_emb ${set_emb} --lam_end ${lam_end} --labaled_size ${labeled_size} --save 0 --learned_loss 1 --proj_name EVAL"
    
    echo "${command}"
    
    # Run the command using sbatch
    sbatch <<< "#!/bin/bash
#SBATCH --job-name=my_experiment_${model}_${run_id}_${lam_end}_${emb_mode}
#SBATCH --output=outputs_reports/eval_output_${model}_${run_id}_${lam_end}_${emb_mode}_%j.txt
#SBATCH --partition=nlp
#SBATCH --account=nlp
#SBATCH --gres=gpu:1
${command}"
}


run_ids=(0 17 26 65)
lab=100
# Run each case with 5 different run_ids
for run_id in "${run_ids[@]}"; do
  ## BERT
  
    #bios
    run_experiment-moji "bert" "$run_id" -1 4 0 "$lab" bios

    #moji
    run_experiment-moji "bert" "$run_id" -1 4 0 "$lab" moji
 
  
done
# Wait for all background processes to finish
wait
