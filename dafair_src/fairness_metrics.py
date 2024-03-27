import torch
import json
from allennlp.fairness import Independence, Separation, Sufficiency  
import numpy as np
def dictionary_torch_to_number(dictionary):
    # Convert torch tensors in the dictionary to numbers
    for key, value in dictionary.items():
        dictionary[key] = value.item() if isinstance(value, torch.Tensor) else value

def get_gap_sum(gap):
    return np.abs(np.array(gap)).sum()

# Loop over different run_id values
run_ids = []  
metric_results = []
for run_id in run_ids:
    if False:
      path = f"models/bios/deberta/pd/{run_id}"  # Update the path
      y_test = torch.load(f"{path}/y.pt")
      y_test_gender = torch.load(f"{path}/z.pt")
      y_test_preds = torch.load(f"{path}/y_pred.pt")
    if True:
      path = f"other_methods/models/jtt/bios/deberta/{run_id}"
      y_test = np.load(f"{path}/y.npy")
      y_test_gender = np.load(f"{path}/z.npy")
      y_test_preds = np.load(f"{path}/y_pred.npy")
      
    unique_labels = len(np.unique(y_test))
    

    # Assuming y_test, y_test_gender, and y_test_preds are NumPy arrays
    y_test = torch.tensor(y_test)
    y_test_gender = torch.tensor(y_test_gender)
    y_test_preds = torch.tensor(y_test_preds)
    
    y_test_preds = y_test_preds.cpu()
    y_test_gender = y_test_gender.cpu()
    y_test = y_test.cpu()

    # Calculate metrics
    independence = Independence(unique_labels, 2, dist_metric="kl_divergence")
    independence(y_test_preds, y_test_gender)
    independence_score = independence.get_metric()

    separation = Separation(unique_labels, 2, dist_metric="kl_divergence")
    separation(y_test_preds, y_test, y_test_gender)
    separation_score = separation.get_metric()

    sufficiency = Sufficiency(unique_labels, 2, dist_metric="kl_divergence")
    sufficiency(y_test_preds, y_test, y_test_gender)
    sufficiency_score = sufficiency.get_metric()
    
    dictionary_torch_to_number(independence_score)
    dictionary_torch_to_number(separation_score)
    dictionary_torch_to_number(sufficiency_score)

    separation_gaps = [scores[0] - scores[1] for label, scores in sorted(separation_score.items())]
    sufficiency_gaps = []
    for label, scores in sorted(sufficiency_score.items()):
      gap = scores[0] - scores[1]
      if not torch.isnan(gap):
          sufficiency_gaps.append(gap.item())
      else:
          sufficiency_gaps.append(0)
    #print('ind',independence_score)
    #print('sep',separation_score)
    #print('suf',sufficiency_score)
    metric_result = {
        #"independence": json.dumps(independence_score),
        #"separation": json.dumps(separation_score),
        #"sufficiency": json.dumps(sufficiency_score),
        "independence_sum": independence_score[0] + independence_score[1],
        "separation_gap-abs_sum": get_gap_sum(separation_gaps),
        "sufficiency_gap-abs_sum": get_gap_sum(sufficiency_gaps)
    }
    print(metric_result)
    metric_results.append(metric_result)
    
# Calculate average over runs
average_metrics = {}
std_metrics = {}

for metric_name in metric_results[0]:  # Assuming all metric results have the same keys
    metric_values = [result[metric_name] for result in metric_results]
    average_metrics[metric_name] = sum(metric_values) / len(metric_values)
    std_metrics[metric_name] = np.std(metric_values)
print("Average metrics over runs:")
print(average_metrics)
print("std metrics over runs:")
print(std_metrics)