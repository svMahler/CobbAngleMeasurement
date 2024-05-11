import os
from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from helper_functions.helper_func import get_image_name

low_text = "low (angle <= 10°)"
mid_text = "mid (10° < angle <= 20°)"
high_text = "high (20° < angle)"

def get_img(img_path, input_path):
    """Get the image from the given path."""
    img_name = get_image_name(img_path)
    path = os.path.join(input_path, img_name)
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return None
    if Path(path).suffix == ".dcm":
        dcm_data = pydicom.dcmread(path)
        image = dcm_data.pixel_array.astype(float)
        rescaled_image = (np.maximum(image, 0) / image.max()) * 255
        final_image = np.uint8(rescaled_image)
        return final_image

    return mpimg.imread(path)
    
def get_normal_img(img_path, output_path):
    """Get the image with drawn normals from the given path.
    Args:
        img_path: Path of the original image.
        output_path: Path of the results directory."""
    img_name = Path(img_path).stem + '_normal.jpg'
    path = os.path.join(output_path, "normals_img", img_name)
    return mpimg.imread(path)
    
def get_predictions_img(img_path, output_path):
    """Get the image with drawn predictions from the given path.
    Args:
        img_path: Path of the original image.
        output_path: Path of the results directory."""
    img_name = Path(img_path).stem + '_predictions.jpg'
    path = os.path.join(output_path, "predictions_img", img_name)
    return mpimg.imread(path)

def get_row(img_path, results, input_path):
    img_name = Path(img_path).stem +".jpg"
    path = os.path.join(input_path, img_name)
    return results[results['image_path'] == path]

def get_diff(eval_results, sets):
    """Calculate the absolute and relative differences for the given sets.
    Args:
        eval_results: DataFrame with the evaluation results.
        sets: List of tuples with the expected, predicted and label column names.
    """
    for label, exp, pred in sets:
        abs_label = "abs_" + label
        eval_results[abs_label] = abs(eval_results[pred] - eval_results[exp])
        eval_results[label] = eval_results[pred] - eval_results[exp]
        eval_results[label] = eval_results.apply(lambda x: x[label] if x[exp] != -1 else np.nan, axis=1)
        eval_results[abs_label] = eval_results.apply(lambda x: x[abs_label] if x[exp] != -1 else np.nan, axis=1)
    return eval_results


def highest_values(output_path, results, input_path, sort_after, exp_label, pred_label):
    """Show the images with the highest differences for PT, MT and TL angle."""
    output_num = 3
    pt_diff_sort = results.sort_values(by=sort_after, ascending=False).head(output_num)

    for index, row in pt_diff_sort.iterrows():
        img_path = row['image_path']
        print(f"Expected: {row[exp_label]}, Predicted: {row[pred_label]} \n Image: {img_path}")
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        plt.figure(figsize=(20, 15))
        axs[0].imshow(get_img(img_path, input_path), cmap='gray')
        axs[0].axis('off')
        axs[0].set_title('Image')

        axs[1].imshow(get_predictions_img(img_path, output_path))
        axs[1].axis('off')
        axs[1].set_title('Predictions')

        axs[2].imshow(get_normal_img(img_path, output_path))
        axs[2].axis('off')
        axs[2].set_title('Normals')
        plt.show()

def get_img_values(output_path, results, img_path, input_path):
    """Show image, predictions and normals for a given image."""
    row = results[results['image_path'] == img_path]
    display(row)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    plt.figure(figsize=(20, 15))
    axs[0].imshow(get_img(img_path, input_path), cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('Image')

    axs[1].imshow(get_predictions_img(img_path, output_path))
    axs[1].axis('off')
    axs[1].set_title('Predictions')

    axs[2].imshow(get_normal_img(img_path, output_path))
    axs[2].axis('off')
    axs[2].set_title('Normals')
    plt.show()
        
def show_histograms(results):
    """Show histograms for PT, MT and TL angle differences. Drops NaN values."""
    bins = range(-100, 100, 2)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.hist(results['pt_diff'].dropna(), bins)
    plt.xlabel('predictions - expectations')
    plt.ylabel('number of samples')
    plt.title('PT angle')
    plt.subplot(1, 3, 2)
    plt.hist(results['mt_diff'].dropna(), bins)
    plt.xlabel('predictions - expectations')
    plt.ylabel('number of samples')
    plt.title('MT angle')
    plt.subplot(1, 3, 3)
    plt.hist(results['tl_diff'].dropna(), bins)
    plt.xlabel('predictions - expectations')
    plt.ylabel('number of samples')
    plt.title('TL angle')
    plt.show()

def show_hist_per_diff(results, exp_label, diff_label, title):
    """Show histograms for low, mid and high angle differences."""
    low_pt = results[(0 <= results[exp_label]) & (results[exp_label] <= 10)]
    mid_pt = results[(10 < results[exp_label]) & (results[exp_label] <= 20)]
    high_pt = results[(20 < results[exp_label])]
    bins = range(-100, 100, 2)

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.hist(low_pt[diff_label], bins)
    plt.xlabel('predictions - expectations')
    plt.ylabel('number of samples')
    plt.title(low_text)
    plt.subplot(1, 3, 2)
    plt.hist(mid_pt[diff_label], bins)
    plt.xlabel('predictions - expectations')
    plt.ylabel('number of samples')
    plt.title(mid_text)
    plt.subplot(1, 3, 3)
    plt.hist(high_pt[diff_label], bins)
    plt.xlabel('predictions - expectations')
    plt.ylabel('number of samples')
    plt.title(high_text)
    plt.suptitle(title, fontsize=20)
    plt.show()

def show_scatter_plot(results, title):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    cleaned_pt = results[results['exp_angle_pt'] != -1]
    cleaned_mt = results[results['exp_angle_mt'] != -1]
    cleaned_tl = results[results['exp_angle_tl'] != -1]
    axs[0].scatter(cleaned_pt['exp_angle_pt'], cleaned_pt['angle_pt'])
    axs[1].scatter(cleaned_mt['exp_angle_mt'], cleaned_mt['angle_mt'])
    axs[2].scatter(cleaned_tl['exp_angle_tl'], cleaned_tl['angle_tl'])
    # Titles
    axs[0].set_title('PT')
    axs[1].set_title('MT')
    axs[2].set_title('TL')
    axs[0].set_xlabel('Expected')
    axs[1].set_xlabel('Expected')
    axs[2].set_xlabel('Expected')
    axs[0].set_ylabel('Predicted')
    fig.suptitle(title)
    plt.show()





