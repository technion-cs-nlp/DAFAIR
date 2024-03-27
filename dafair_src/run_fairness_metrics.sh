#!/bin/bash

#SBATCH --job-name=fair
#SBATCH --output=outputs_reports/eval_bert-bios-jtt-%j.txt
#SBATCH --partition=nlp
#SBATCH --account=nlp
#SBATCH --gres=gpu:1  
python fairness_metrics.py