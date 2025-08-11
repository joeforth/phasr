# YOLOv11 Phase Detection in AOT/Oil/Water Experiments

This project demonstrates the fine-tuning of a **YOLOv11** object detection model to identify and classify **phases in an AOT/Oil/Water experiment**, 
including the ability to **calculate phase sizes**.

## Project Overview

- **Objective:** Detect and classify distinct phases (e.g., AOT, oil, water) in experimental setups involving surfactants and liquids.
- **Model:** YOLOv11 (fine-tuned)
- **Additional Processing:** Post-inference phase size estimation based on detected bounding boxes.

## Experiment Context

The AOT/Oil/Water system is a classic three-phase system studied in soft matter and colloidal science. The fine-tuned YOLOv11 model helps:

- Automatically detect phase boundaries.
- Quantify the relative sizes of each phase.

##  Model Training

- **Base Model:** YOLOv11 pre-trained on COCO
- **Custom Dataset:** Manually labelled images of the AOT/oil/water system using bounding boxes with the roboflow tool
- **Classes:**
  Vial  
  foam  
  slightly turbid  
  transparent  
  very turbid
-**Framework** The model was trained using Ultralytics pip install

###  Download the Model

Download model weights (https://drive.google.com/drive/folders/1HV_qvKs7KJxT8U7x1opjqCJlEw4spbQJ?usp=sharing)


## How to run the project

  Clone the repository and add the model to the folder.
  Run the main.py
  Select a video source
  Place the vial(s) between 5 and 20 cm for best results.
  Set up the experiment by choosing a folder to store the results.
  Either configure the timer for automatic inference or click the capture and detect button to run the inference manually.


The use of YOLO from Ultralytics was made under the AGPL-3.0 License.
