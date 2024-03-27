
run_ids=(0 17 26 57 65)
for run_id in "${run_ids[@]}"; do
    sbatch --gres=gpu:1 --wrap="python -u extract_vectors_moji.py --run_id $run_id --dataset bios" --output=outputs_reports/ext_bios_%j.txt --partition=nlp --account=nlp
    sbatch --gres=gpu:1 --wrap="python -u extract_vectors_moji.py --run_id $run_id --dataset moji" --output=outputs_reports/ext_moji_%j.txt --partition=nlp --account=nlp
done
