import os
from pathlib import Path

import cv2
import pandas as pd

import helper_functions.calc as calc
import helper_functions.draw as draw
import helper_functions.helper_func as hf

FLAG_MEANING = {0: 'Everything ok',
                1: 'Not increasing x_values',
                2: 'Not enough points for calculating normals',
                3: 'rad2deg/arccos could not be calculated'}

def processing(pred_path, output_path, error_path, params, use_images=True):
    results_path = os.path.join(output_path, 'results.csv')

    detection_list = hf.load_predictions(pred_path, error_path)
    print(f"Processing predictions for {len(detection_list)} images...")
    for image_path, bbox_list in detection_list.items():
        if use_images:
            image = cv2.imread(image_path)
        else:
            image = None
        
        if len(bbox_list) < 2:
            hf.write_error(error_path, image_path, None, "Error: Only 2 vertebrae detected")
            continue

        all_centroids_list = hf.get_centroids(bbox_list) 
        centroids_list , bbox_list = hf.remove_outliers(all_centroids_list, bbox_list)
        if len(centroids_list) < 4:
            hf.write_error(error_path, image_path, None, "Error: less than 4 vertebrae were accepted")
            continue

        res_dict, normal_image, flag = calc.calculate_values(image_path, image, centroids_list, params, use_images)
        if use_images:
            img_predict = draw.draw_bboxes(image, bbox_list)
            img_predict = draw.draw_points(img_predict, centroids_list)

            normal_img_name = Path(image_path).stem + '_normal.jpg'
            normal_dir = os.path.join(output_path, 'normals_img')
            normal_path = os.path.join(normal_dir, normal_img_name)

            predict_img_name = Path(image_path).stem + '_predictions.jpg'
            predict_dir = os.path.join(output_path, 'predictions_img')
            predict_path = os.path.join(predict_dir, predict_img_name)

            save_image(normal_path, normal_image, res_dict, error_path)
            save_image(predict_path, img_predict, res_dict, error_path)
        
        if len(centroids_list) < 15:
            res_dict["info"] = f"Warning: only {len(centroids_list)} vertebrae were detected"

        if flag != 0:
            hf.write_error(error_path, image_path, res_dict, 
                           f"Something went wrong while calculating the values: {FLAG_MEANING[flag]}")
        elif res_dict is None:
            hf.write_error(error_path, image_path, res_dict, "Could not calculate values.")
        else:
            res_df = pd.DataFrame([res_dict])
            res_df.to_csv(results_path, mode ='a', header= not os.path.exists(results_path), index=False)

    print("Done")
    return results_path

def save_image(path, img, values, error_path):
    try:
        success = cv2.imwrite(path, img)
        if not success:
            print(f"Could not save image: {path}")
            hf.write_error(error_path, path, values, "Could not save image.")
    except Exception as e:
        print(f"Could not save image: {path}")
        print(f"Exception: {e}")
        hf.write_error(error_path, path, values, f"Could not save image: {e}")
    






