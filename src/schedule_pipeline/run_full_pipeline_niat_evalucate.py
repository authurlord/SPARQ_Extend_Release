import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluator import Evaluator, niat_match_func_for_samples



if __name__ == "__main__":
    path = '/home/yanmy/SPARQ/schedule_pipeline/tmp/niat_test_4B/preds_and_golds.json'
    with open(path, 'r') as f:
        all_samples = json.load(f)
            
    with open("../datasets/NIAT/niat_4000_filtered.json", 'r') as f:
        data = json.load(f)

    with open("../datasets/NIAT/niat_4000_filtered_token_stats.json") as f:
        token_ranges = json.load(f)

    range_distribution = {}
    for key, ranges in token_ranges['range_distribution'].items():
        range_distribution[key] = {
            'idx': [],
            'samples': [],
        }
        for _, pair in enumerate(ranges):
            idx = pair['idx']
            table_id = pair['table_id']
            answer = pair['answer']
            range_distribution[key]['idx'].append(idx)
            range_distribution[key]['samples'].append(all_samples[idx])
    


    print("Start evaluating")
    for key, info in range_distribution.items():
        samples = info['samples']
        acc_all = niat_match_func_for_samples(samples, strategy="top") * 100
        print(key, f"Accuracy (NIAT EM): {acc_all:.2f}%")