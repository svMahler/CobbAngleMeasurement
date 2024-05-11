import datetime
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from matplotlib import pyplot as plt



def get_centroids(bbox_list, show=False):
    """ Returns a list of centroids from a list of bounding boxes.
    Args:
        bbox_list: list of bounding boxes in the form [x1, y1, x2, y2]
        show: whether to show the plot
    """
    centroids_list = []
    for bbox in bbox_list:
        x1, y1, x2, y2 = bbox
        x = x1 + (x2-x1) / 2
        y = y1 + (y2-y1) / 2
        centroids_list.append([x, y])
    
    # draw function
    if show:
        # changed x and y values for drawing
        x_values = [x[1] for x in centroids_list]
        y_values = [x[0] for x in centroids_list]
        plt.plot(x_values, y_values, 'o')
        plt.title('Centroids')
        plt.show()
    return centroids_list

def remove_outliers(centroids_list, bbox_list):
    """Removes outliers from a list of points. """
    sorted_list = [[centroid, bbox] for centroid, bbox in 
                   sorted(zip(centroids_list, bbox_list), key=lambda x: x[0][1])] #sorted after y-value of centroid
    filtered_centroids = []
    filtered_bbox = []
    c = sorted_list[0][0]
    b = sorted_list[0][1]
    half_width = abs(b[2]-b[0]) /2
    bottom = sorted_list[1][0]
    if (abs(bottom[0]-c[0]) < half_width):
        filtered_centroids += [c]
        filtered_bbox += [b]

    for i in range(1, len(sorted_list) -1):
        c = sorted_list[i][0]
        b = sorted_list[i][1]
        half_width = (b[2]-b[0]) /2
        bottom = sorted_list[i+1][0] # bottom = higher y_value than top
        top = sorted_list[i-1][0]
        if (abs(bottom[0]-c[0]) < half_width) | (abs(c[0] - top[0]) < half_width):
            filtered_centroids += [c]
            filtered_bbox += [b]
            
    c = sorted_list[-1][0]
    b = sorted_list[-1][1]
    half_width = (b[2]-b[0]) /2
    top = sorted_list[-2][0]
    if (abs(c[0] - top[0]) < half_width):
        filtered_centroids += [c]
        filtered_bbox += [b]
    
    #print(f"Removed outliers, Before:{len(centroids_list)}, Afterwards: {len(filtered_centroids)}")
    return filtered_centroids, filtered_bbox

def load_predictions(path, error_path):
    """Loads the predictions from a json file. Removes image if nothing was predicted, writes to error file."""
    with open(path) as json_data:
        data = json.load(json_data)

    df_annotations = pd.DataFrame(data['annotations'])
    if df_annotations.empty:
        print("Error: something went wrong, nothing predicted")
        return {}

    image_dict = {}
    for image in data['images']:
        image_id = image['id']
        image_ann = df_annotations[
            (df_annotations['image_id'] == image_id) & (df_annotations['category_id'] == 0)]

        if image_ann.empty:
            write_error(error_path, image['file_name'], None, 'Nothing predicted for this image')
            continue

        bbox_list = image_ann['bbox'].tolist()
        if len(bbox_list) > 0:
            image_dict[image['file_name']] = bbox_list
        else:
            write_error(error_path, image['file_name'], None, 'No bounding boxes found for this image')

    return image_dict

def construct_tree(output_dir):
    """Constructs the directory tree for the results.
    output_dir
    ├── date_results
    │   ├── normals_img
    │   ├── predictions_img
    │   ├── error.csv

    Args:
        output_dir: path to the output directory
    Returns:
        results_dir: path to the results directory
    """
    if not Path(output_dir).exists():
        os.makedirs(output_dir, exist_ok=True)

    current_datetime = datetime.datetime.now()
    date_string = current_datetime.strftime("%Y-%m-%d")

    results_dir = os.path.join(output_dir, "{}_results".format(date_string))
    images_dir = os.path.join(results_dir, 'normals_img')
    images_boxes_dir = os.path.join(results_dir, 'predictions_img')
    
    if Path(results_dir).exists():
        print(f"Error: {results_dir} already exists. Please rename it or delete it manually.")
        return None, None
    os.makedirs(results_dir, exist_ok=True)
    
    if not Path(images_dir).exists():
        os.makedirs(images_dir, exist_ok=True)
    if not Path(images_boxes_dir).exists():
        os.makedirs(images_boxes_dir, exist_ok=True)

    error_path = os.path.join(results_dir, 'error.csv')
    error_df = pd.DataFrame(columns=['image_path', 'error', 'values'])
    error_df.to_csv(error_path, mode='a', header=True, index=False)

    return results_dir, error_path

def transform_input(input_path, error_path):
    """Transforms the input images to the correct format."""
    all_files = glob.glob(os.path.join(input_path, '*.*'), recursive=True)
    new_input_path = os.path.join(input_path, "tmp_ca")
    if Path(new_input_path).exists():
        print(f"Error: {new_input_path} already exists. Please rename it or delete it manually.")
        return None
    os.makedirs(new_input_path, exist_ok=True)
    path_list = []

    for file in all_files:
        file_path = os.path.join(input_path, file)
        if os.path.isfile(file_path):
            file_type = Path(file_path).suffix
            if file_type == ".jpg":
                path_list.append(file_path)
            elif file_type == ".dcm":
                path_list.append(handle_dicom_file(file_path, new_input_path))
            elif file_type == ".png":
                path_list.append(handle_png_file(file_path, new_input_path))
            else:
                print(f"Error: wrong file type: {file_type}; Expected: .jpg, .dcm, .png")
                write_error(error_path, file_path, None, f"Wrong file type: {file_type}")
        else:
            print(f"Error: Folders are not processed: {file_path}")
            write_error(error_path, file_path, None, "Folders are not processed")

    print(f"Found {len(path_list)} images in {input_path}")
    return path_list, new_input_path
        
def handle_dicom_file(file_path, input_path):
    """Handles the dicom file and saves it as a jpg image."""
    new_file_path = os.path.join(input_path, get_image_name(file_path) + ".jpg")
    dcm_data = pydicom.dcmread(file_path)
    image = dcm_data.pixel_array.astype(float)
    rescaled_image = (np.maximum(image,0)/image.max())*255
    final_image = np.uint8(rescaled_image)
    final_image = Image.fromarray(final_image)
    final_image.save(new_file_path)
    return new_file_path
 
def handle_png_file(file_path, input_path):
    """Handles the png file and saves it as a jpg image."""
    image = Image.open(file_path)
    new_file_path = os.path.join(input_path, get_image_name(file_path) + ".jpg")
    image = image.convert('RGB')
    image.save(new_file_path, 'JPEG')
    return new_file_path

def get_image_name(image_path):
    """Returns the original image path."""
    dcm_ending = ".dcm.jpg"
    png_ending = ".png.jpg"
    if image_path.endswith(dcm_ending) or image_path.endswith(png_ending):
        return Path(image_path).stem
    else:
        return Path(image_path).name


def write_error(error_path, image_path, values, error):
    """Writes the error to the error_path csv file."""
    name = get_image_name(image_path)
    data = {'image_path': name, 'error': error, 'values': values}
    df = pd.DataFrame([data])

    with open(error_path, 'a') as f:
        df.to_csv(f, header=False, index=False)


def get_COCO_format():
    """Returns the COCO format."""
    dict = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
    }
    dict['categories'] += [{"id": 0,
                            "name": "vertebra",
                            "supercategory": "spine",
                            }]
    return dict











