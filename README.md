# Chest abnormalities detection

[![License](http://img.shields.io/badge/license-MIT-green.svg?style=flat)](https://github.com/Solvve/ml_chest_xray_issue_classifaction/blob/master/LICENSE)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-378/)
[![scikit-learn 0.23.2](https://img.shields.io/badge/scikit_learn-0.23.2-blue)](https://scikit-learn.org/stable/)
[![Solvve](https://img.shields.io/badge/made%20in-solvve-blue)](https://solvve.com/)

## Description
Detection of abnormalities on chest X-ray images.

We need to find several abnormalities on chest X-ray images.

Example of images

<img src="images/VinBigDataChestXray.png"
     alt="example"
     style="float: left; margin-right: 10px;" />

For solving this problem we will apply FasterRCNN, Yolo_v5 using PyTorch and PyTorch_Lightning libraries 

And additionally we will build binary classifier to classify image - normal/abnormal

As a result of our work we will build simple web app using Flask which is capable of taking image and detecting abnormalities on it.

## Installation yolo v5 run

For install Yolo v5 run following commands:

1. !git clone https://github.com/ultralytics/yolov5
2. !mv yolov5/* ./
3. !pip install -r requirements.txt

## Analysis and modeling
We follow the next steps:
1. EDA (notebooks/EDA.ipynb)
3. Modeling : Yolo v5 (notebooks/Yolov5.ipynb)
4. Modeling : FasterRCNN (notebooks/FasterRCNN.ipynb)
5. Modeling : binary classifier - abnormal/normal image (notebooks/binary_classifier.ipynb)

## Datasets 

1. https://www.kaggle.com/awsaf49/vinbigdata-512-image-dataset

2. https://www.kaggle.com/raddar/vinbigdata-competition-jpg-data-2x-downsampled

3. https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data

## Web app for abnormaly detection

Example of web app for abnormalities detection

Step 1. Uploading image

<img src="images/upload.png"
     alt="upload"
     style="float: left; margin-right: 10px;" />

Step 2. Click submit and see the result

<img src="images/results_1.png"
     alt="upload"
     style="float: left; margin-right: 10px;" />

<img src="images/results_2.png"
     alt="upload"
     style="float: left; margin-right: 10px;" />

<img src="images/results_3.png"
     alt="upload"
     style="float: left; margin-right: 10px;" />


