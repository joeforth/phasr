# YOLOv11 STFA Phase Detection in AOT/Oil/Water Experiments

This project demonstrates the  training of a new architecture for Video Instance segmentation by combining **YOLO11** and the new **STFA block** from Duran et.al. from the paper "Video object detection via space-time feature aggregation and result reuse model" to detect, segment and classify **phases in an AOT/Oil/Water experiment**, as well as track them to have a more realistic video instance segmentation.

## Project Overview

- **Objective:** Detect, segment and classify distinct phases (e.g., AOT, oil, water) in experimental setups involving surfactants and liquids.
- **Model:** YOLOv11 STFA (fine-tuned)
- **Additional Processing:** Tracking of the results for a more consistent segmentation and detection of the phases.

## Experiment Context

- Automatically detect and segment phase boundaries.
- Keep track of detected phases by matching the middle point of each bounding box and averaging the masks to create a more accurate
  segmentation mask over time of the phases detected.


##  Model
![YOLO11 STFA architechture](/assets/YOLO11STFAARCH.jpg)
- **Base Model:** YOLOv11 with no SPPF and C2PSA blocks instead the Space-Time-Feature agregation block was added to replace these to introduce
  the temporal aspect when performing video instance segmentation
- **Custom Dataset:** Manually labelled images of the AOT/oil/water system using bounding boxes with the roboflow tool
- **Classes:**
  Vial  
  foam  
  slightly turbid  
  transparent  
  very turbid
-**Framework** The model was trained using Ultralytics model.train() with a modified neural network folder to accept the changes made.
               The input of the model is 3 frames concatenated, resulting in a shape of (640,480,9)
               To run the model with one image, concatenate the image with itself. 


###  Download the Model

Download model weights (https://drive.google.com/drive/folders/1HV_qvKs7KJxT8U7x1opjqCJlEw4spbQJ?usp=sharing)

### Files

GUI.py, RR.py and Tracker.py need to be together to run the GUI file
RR.py contains the code to run the model and output the annotated frame.
Tracker.py tracks and smooths the mask to perform a better detection.
GUI.py contains the code for the GUI
yolo11n-segALI.yaml is the yaml file that builds the model. Reading it gives clues as to how the model is built and what blocks or modules make up the model.


## How to run the project

  Clone the repository and add the model to the folder. 
  run pip install PyQt5 opencv-python ultralytics numpy scipy pygrabber
  go to the location of the ultralytics install (pip show ultralytics) and change the nn folder to the one in the repository
  Run the GUI.py
  Select a video source
  Place the vial(s) between 5 and 20 cm for best results.
  start tracking



The use of YOLO from Ultralytics was made under the AGPL-3.0 License.
