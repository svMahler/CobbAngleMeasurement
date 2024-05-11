import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from helper_functions.draw import draw_angle_to_value


def transform(results):
    """Transforms the results dataframe to the correct dataformat."""
    results['exp_angle_tl'] = pd.to_numeric(results['exp_angle_tl'], errors='coerce').fillna(-1)
    results['exp_angle_mt'] = pd.to_numeric(results['exp_angle_mt'], errors='coerce').fillna(-1)
    results['exp_angle_pt'] = pd.to_numeric(results['exp_angle_pt'], errors='coerce').fillna(-1)
    results['centroids'] = results['centroids'].apply(lambda x: eval(x))
    results['normals'] = results['normals'].apply(lambda x: eval(x))
    results['points'] = results['points'].apply(lambda x: eval(x))
    results['min_x'] = results['min_x'].astype(int)
    results['max_x'] = results['max_x'].astype(int)
    results['info'] = results['info'].fillna("")

    return results


def calculate_metrics_three_angle(results, dev=False):
    """Calculates the metrics for the three angles and returns them in a dictionary."""

    header = [('exp_angle_tl', 'angle_tl', 'tl'), ('exp_angle_mt', 'angle_mt', 'mt'),
              ('exp_angle_pt', 'angle_pt', 'pt')]
    values_dict = {}
    for exp, pred, label in header:
        dropped = results[results[exp] != -1]  # remove rows without an expected value
        mse_angle = mean_squared_error(dropped[exp], dropped[pred])
        smape_angle = smape_one_angle(dropped[exp], dropped[pred])
        mae_angle = mean_absolute_error(dropped[exp], dropped[pred])
        r2_angle = r2_score(dropped[exp], dropped[pred])
        values_dict[label] = {"SMAPE": smape_angle,
                              "MSE": mse_angle,
                              "MAE": mae_angle,
                              "R2": r2_angle}
    if dev:
        smape_angle = smape_all_angles(results)
        values_dict["all_angles"] = {"SMAPE": smape_angle, "MSE": "None", "MAE": "None", "R2": "None"}

        # SRI, SCD
        sum_exp_angle = results[results != -1][['exp_angle_tl', 'exp_angle_mt', 'exp_angle_pt']].sum(axis=1)
        mae_sri = mean_absolute_error(results['sri'], sum_exp_angle)
        mae_scd = mean_absolute_error(results['scd'], sum_exp_angle)
        r2_sri = r2_score(results['sri'], sum_exp_angle)
        r2_scd = r2_score(results['scd'], sum_exp_angle)
        values_dict["sri"] = {"SMAPE": "None", "MSE": "None", "MAE": mae_sri, "R2": r2_sri}
        values_dict["scd"] = {"SMAPE": "None", "MSE": "None", "MAE": mae_scd, "R2": r2_scd}

        plt.figure(figsize=(15, 8))
        draw_angle_to_value(results['sri'], sum_exp_angle, "SRI", 'Expected Cobb Angle to SRI', 1)
        draw_angle_to_value(results['scd'], sum_exp_angle, "SCD", 'Expected Cobb Angle to SCD', 2)
    else:
        results_greatest_angle = results[['angle_pt', 'angle_mt', 'angle_tl']].max(axis=1)
        smape_greatest = smape_one_angle(results['exp_greatest_angle'], results_greatest_angle)
        mse_greatest = mean_squared_error(results['exp_greatest_angle'], results_greatest_angle)
        mae_greatest = mean_absolute_error(results['exp_greatest_angle'], results_greatest_angle)
        r2_greatest = r2_score(results['exp_greatest_angle'], results_greatest_angle)
        values_dict["greatest_angle"] = {"SMAPE": smape_greatest, "MSE": mse_greatest, "MAE": mae_greatest,
                                         "R2": r2_greatest}

    return values_dict


def remove_not_enough_detected(results, limit):
    """Removes results with less than limit detected vertebrae"""

    print(f"Removing results with less than {limit} detected vertebrae")
    numbers_df = pd.DataFrame(index=range(0, len(results.index)))
    pattern = r'(\d+)'
    numbers_df['num'] = results['info'].str.extract(pattern, expand=False)
    numbers_df['num'] = pd.to_numeric(numbers_df['num'], errors='coerce').fillna(limit + 1)
    before = results.shape[0]
    results = results[numbers_df['num'] >= limit]
    after = results.shape[0]
    print(f"Removed {before - after} results with less than {limit} detected vertebrae")
    return results


def load_results(label_path, results_path, exp_angle_opt=False, set_zero=False):
    """Loads the expected results and the results from the results_path and returns the merged dataframe"""

    images_angles, dev = load_expected_results(label_path)
    results = pd.read_csv(results_path)
    images_angles = images_angles.merge(results, on='image_path')

    num_before = images_angles.shape[0]
    print(f"Loaded {num_before} results")

    if exp_angle_opt:
        images_angles = remove_errors_dataset(images_angles)
    if set_zero:
        images_angles = set_to_zero(images_angles)

    return images_angles, dev


def load_expected_results(label_path):
    """Loads the expected results from the label_path and returns the dataframe.
    Checks if the dataset is a verification or a development/validation dataset.
    """
    expectations_path = os.path.join(label_path, 'expectations.csv')
    filenames_path = os.path.join(label_path, 'filenames.csv')
    angles_path = os.path.join(label_path, 'angles.csv')
    if os.path.exists(expectations_path):
        print("Expectations file found: Verification dataset format is used.")
        headers_labels = ['image_path', 'exp_angle_mt', 'exp_angle_pt', 'exp_angle_tl', 'mt_orientation',
                          'pt_orientation', 'tl_orientation', 'exp_greatest_angle']
        images_angles = pd.read_csv(expectations_path, header=None)
        images_angles.columns = headers_labels
        dev = False
    elif os.path.exists(filenames_path) and os.path.exists(angles_path):
        print("Expectations file not found: Development and validation dataset format is used.")
        headers_labels = ['image_path', 'exp_angle_mt', 'exp_angle_pt', 'exp_angle_tl']
        label_images = pd.read_csv(filenames_path, header=None)
        label_angles = pd.read_csv(angles_path, header=None)
        images_angles = pd.concat([label_images, label_angles], axis=1, ignore_index=True)
        images_angles.columns = headers_labels
        dev = True
    else:
        print(
            "ERROR: Wrong label_path. For validation/development dataset, please provide path to filenames.csv and "
            "angles.csv. For verification dataset, please provide path to expectations.csv.")
        exit()
    return images_angles, dev


def remove_errors_dataset(df):
    """Removes rows with an expected angle < 1"""
    false = df[(0.0 < df['exp_angle_pt']) & (df['exp_angle_pt'] < 1.0) |
               (0.0 < df['exp_angle_mt']) & (df['exp_angle_mt'] < 1.0) |
               (0.0 < df['exp_angle_tl']) & (df['exp_angle_tl'] < 1.0)]

    print(f"Removed {len(false)} results because of weird expected values")
    return df[~df.index.isin(false.index)]


def set_to_zero(df):
    """Sets the values to 0 if they are < 4"""
    columns = ['angle_pt']
    for column in columns:
        df[column] = df[column].apply(lambda x: 0 if x < 4 else x)
    return df


def smape_all_angles(results):
    """Calculates the smape for all angles"""
    value = 0
    for i, row in results.iterrows():
        value += ((abs(row['angle_tl'] - row['exp_angle_tl'])
                   + abs(row['angle_mt'] - row['exp_angle_mt'])
                   + abs(row['angle_pt'] - row['exp_angle_pt']))
                  / (row['angle_tl'] + row['exp_angle_tl'] + row['angle_mt'] + row['exp_angle_mt'] + row['angle_pt'] +
                     row['exp_angle_pt']))

    return (value * 200) / len(results)


def smape_one_angle(y_true, y_pred):
    """Calculates the smape for one angle"""

    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)

    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
