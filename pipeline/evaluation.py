import os

import matplotlib.pyplot as plt
import json
from helper_functions.eval import transform, calculate_metrics_three_angle, remove_not_enough_detected, load_results

"""This file is used to calculate the metrics for the given results"""


def evaluate(results_path, label_path, output_path, limit=0, exp_angle_opt=False, set_zero=False):
    """Calculates the metrics for the given results
    Excluded results are:
        - results with less than limit detected vertebrae
        - results with no expected values
        - results with expected angles < 1 (optional)

        Args:
            results_path: path to the results file
            label_path: path to the labels file
            output_path: path to the output directory
            limit: minimum number of detected vertebrae
    """
    print("Starting Evaluation...")
    results, dev = load_results(label_path, results_path, exp_angle_opt, set_zero)
    results = transform(results)
    results = remove_not_enough_detected(results, limit)
    

    print("Calculating metrics...")
    values_dict = calculate_metrics_three_angle(results, dev)
    if dev:
        plt.savefig(os.path.join(output_path, "sri_scd_figures.png")) 

    print("Saving metrics...")
    
    filepath = os.path.join(output_path, "metrics.json")
    with open(filepath, 'w') as json_file:
        json.dump(values_dict, json_file, indent=4)

    print("Done!")
    return values_dict


