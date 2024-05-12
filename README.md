# Cobb Angle Measurement via Neural Networks

This repository provides code for calculating the Cobb Angle from X-ray images of the spine. 
The code uses a RetinaNet model of  the detectron2 framework to predict the bounding boxes of the vertebrae in the X-ray images. 
The predicted bounding boxes are then used to calculate the Cobb Angle, the SRI (Spine Regression Integral) and the SCD (Spine Center Distance).

## Usage
### Local
The code can be run locally:
```bash
python main.py -i /path/to/your/image/directory -o /path/to/your/output/directory
```
### Input format
Accepted image formats are: _.png_, _.jpg_, _.dcm_


### Output format
In the output directory a new directory _date_results_ will be created with the following structure:

- _normals_img_: images of the normals used to calculate the Cobb Angle
- _predictions_img_: images of the predicted bounding boxes
- _predictions.json_: the predicted bounding boxes in COCO format
- _results.csv_: the calculated Cobb Angles, SRI, SCD and other information
- _errors.csv_: the errors that occurred during the calculation


<sub>Hint: Install torch before installing _requirements.txt_.</sub>
