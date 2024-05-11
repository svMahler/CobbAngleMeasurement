import json
import os
import torch
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import helper_functions.helper_func as hf

PREDICTIONS_ENDING = "predictions.json"

WEIGHTS_PATH = "./model/model_weights.pth"
CONFIG_FILE = "./model/cfg.yaml"
CONF_THRESHOLD = 0.5


def predict(path_list, output_path, error_path):
    """Predicts the bounding boxes for the given images and stores the results in a json file."""
    predictor = create_predictor()
    data = hf.get_COCO_format()

    for img_count in range(len(path_list)):
        img = cv2.imread(path_list[img_count])
        if img is None: 
            print("Error: Something went wrong while loading the file ", path_list[img_count])
            hf.write_error(error_path, path_list[img_count], None, "Error: Something went wrong while loading the file")
            continue
            
        h, w, _ = img.shape
        data['images'] += [{"id": img_count,
                           "width": w,
                           "height": h,
                           "file_name": path_list[img_count],
                           "license": 0,
                           }]

        predictions = predictor(img)   
        instances = predictions['instances']
        scores = instances.scores 
        num_instances = len(scores)
        

        for pred_count in range(num_instances):
            if scores[pred_count].item() > CONF_THRESHOLD:
                data['annotations'] += [pred_to_annotation(instances, pred_count, img_count, pred_count*img_count)]
                

    return store_predictions(data, output_path)


def pred_to_annotation(instances, pred_id, img_id, ann_id):
    """Converts the prediction to an annotation."""
    pred_boxes = instances.pred_boxes
    scores = instances.scores 
    pred_classes = instances.pred_classes
    return {
        "id": ann_id,
        "image_id": img_id,
        "category_id": 0,
        "bbox": pred_boxes.tensor[pred_id].tolist(),
        "iscrowd": 0,
        "score": scores[pred_id].item(),
    }

def create_predictor():
    """Creates the predictor for the model."""
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_FILE)
    cfg.MODEL.WEIGHTS = WEIGHTS_PATH
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONF_THRESHOLD
    if torch.cuda.is_available():
        print("Using GPU")
        cfg.MODEL.DEVICE = "cuda"
    else:
        print("Using CPU")
        cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    return predictor

def store_predictions(data, output_path):
    """Stores the predictions in a json file in the output_path directory."""

    print("Storing predictions...")
    filename = "{}".format(PREDICTIONS_ENDING)

    os.makedirs(output_path, exist_ok=True)
    path = os.path.join(output_path, filename)
    with open(path, 'w') as outfile:
        json.dump(data, outfile)

    print("Predictions stored in {}".format(path))
    return path