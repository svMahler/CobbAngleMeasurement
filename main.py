import os
import time
import argparse
import json
from pathlib import Path

import helper_functions.helper_func as hf
from pipeline import calculation, evaluation, prediction

PARAMS_PATH = 'model/params.json'
PREDICTIONS_ENDING = "predictions"

def main(input, output, verification, eval):
    start = time.time()
    output_path, error_path = hf.construct_tree(output)
    if not output_path:
        exit()
    tree_time = time.time()

    if verification:
        print("Given landmarks are used!!!")
        pred_path = verification
        tmp_dir = None
    else:
        path_list, tmp_dir = hf.transform_input(input, error_path)
        if not path_list:
            exit()
        pred_path = prediction.predict(path_list, output_path, error_path)

    pred_time = time.time()
        
    params = json.load(open(PARAMS_PATH, "r"))
    
    res_path = calculation.processing(pred_path, output_path, error_path, params)
    calc_time = time.time()

    if eval:
        metrics = evaluation.evaluate(res_path, eval, output_path)
        print("Metrics:", metrics)

    end = time.time()

    if Path(tmp_dir).exists():
        os.rmdir(tmp_dir)

    print(f"Tree creation: {tree_time - start}")
    print(f"Prediction: {pred_time - tree_time}")
    print(f"Calculation: {calc_time - pred_time}")
    print(f"Total: {end - start}")
    time_path = output_path + "/time.txt"
    with open(time_path, "w") as file:
        file.write(f"Tree creation: {tree_time - start}\n")
        file.write(f"Prediction: {pred_time - tree_time}\n")
        file.write(f"Calculation: {calc_time - pred_time}\n")
        file.write(f"Total: {end - start}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict and Calculate the Cobb Angle")
    parser.add_argument("-i", "--input", metavar="DIRECTORY", dest="input",
                        help="Path to directory to be processed.", type=str, required=True)

    parser.add_argument("-o", "--output", metavar="DIRECTORY", dest="output",
                        help="Output directory for results", type=str, required=True)

    parser.add_argument("-eval", "--evaluation", metavar="DIRECTORY", dest="evaluation",
                        help="Directory for expected results", type=str, required=False)

    parser.add_argument("-verify", "--verification", metavar="DIRECTORY", dest="verification",
                        help="Use given landmarks instead of predictions, landmarks directory needed", type=str,
                        required=False)

    args = parser.parse_args()

    input = args.input
    output = args.output
    verification = args.verification
    eval = args.evaluation

    main(input, output, verification, eval)
