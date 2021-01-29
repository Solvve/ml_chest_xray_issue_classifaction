# Chest abnormalities detection

[![License](http://img.shields.io/badge/license-MIT-green.svg?style=flat)](https://github.com/Solvve/ml_gas_stations_forecast/blob/master/LICENSE.txt)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-378/)
[![scikit-learn 0.23.2](https://img.shields.io/badge/scikit_learn-0.23.2-blue)](https://scikit-learn.org/stable/)
[![Solvve](https://img.shields.io/badge/made%20in-solvve-blue)](https://solvve.com/)

## Description
When you have a broken arm, radiologists help save the day—and the bone. These doctors diagnose and treat medical conditions using imaging techniques like CT and PET scans, MRIs, and, of course, X-rays. Yet, as it happens when working with such a wide variety of medical tools, radiologists face many daily challenges, perhaps the most difficult being the chest radiograph. The interpretation of chest X-rays can lead to medical misdiagnosis, even for the best practicing doctor. Computer-aided detection and diagnosis systems (CADe/CADx) would help reduce the pressure on doctors at metropolitan hospitals and improve diagnostic quality in rural areas.

Existing methods of interpreting chest X-ray images classify them into a list of findings. There is currently no specification of their locations on the image which sometimes leads to inexplicable results. A solution for localizing findings on chest X-ray images is needed for providing doctors with more meaningful diagnostic assistance.

Established in August 2018 and funded by the Vingroup JSC, the Vingroup Big Data Institute (VinBigData) aims to promote fundamental research and investigate novel and highly-applicable technologies. The Institute focuses on key fields of data science and artificial intelligence: computational biomedicine, natural language processing, computer vision, and medical image processing. The medical imaging team at VinBigData conducts research in collecting, processing, analyzing, and understanding medical data. They're working to build large-scale and high-precision medical imaging solutions based on the latest advancements in artificial intelligence to facilitate effective clinical workflows.

## Analysis and modeling
We follow the next steps:
1. EDA (notebooks/EDA.ipynb)
3. Modeling : Yolo v5 (notebooks/Yolov5.ipynb)
4. Modeling : FasterRCNN (notebooks/FasterRCNN.ipynb)
5. Modeling : binary classifier - abnormal/normal image (notebooks/binary_classifier)

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



