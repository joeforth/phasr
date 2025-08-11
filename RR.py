#This file can only be run if the correct model is in the same file as the rest please download the YOLO STFA model



import math
import random

import cv2 as cv
from ultralytics import YOLO
import numpy as np
from Tracker import Tracker

# Variable Initialisation
model = YOLO('nb1AUGMENT.pt') #Model curently YOLO STFA input shape is (480, 640, 9)  3 images concatenated
previous = []
cam = cv.VideoCapture(0,cv.CAP_DSHOW) #camera
good = []
COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Teal
]
cont = True
tracker = Tracker(distance_threshold=25.0, smoothing_factor=0.5) # bbox and masks tracker eliminates jitter
memory = []

while cont == True:
    ret, frame = cam.read()
    framesused = []
    cv.imshow("azdaz", frame)
    #Create a frame memorry buffer where past frames can be taken from at the moment it is random but can
    # be made better by using heuristic function to choose the frames
    if len(memory) in range(30):
        memory.append(frame)
    else:
        memory.pop(0)
        memory.append(frame)

    for i in range(2):
        index1 = random.randint(0, len(memory) - 1)
        framesused.append(memory[index1])

    # Convert to NumPy arrays
    img1_arr, img2_arr, img3_arr = np.array(framesused[0]), np.array(framesused[1]), np.array(frame)

    # The shape will now correctly be (480, 640, 9)
    concatenated_array = np.concatenate([img3_arr, img2_arr, img1_arr], axis=2)


    results = model(concatenated_array, conf=0.5) # Run the model curent inference speed on NVIDIA 3070 laptop GPU is 200 ms

    #Post processing
    new = []
    for result in results:
        if result.masks != None and result.boxes != None:
            for mask, box in zip(result.masks, result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(np.int32)
                middleX = math.ceil(x1 + (x2 - x1) / 2)
                middleY = math.ceil(y1 + (y2 - y1) / 2)
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                mask = mask.data.cpu().numpy()
                current_prediction = {
                    'middle': [middleX, middleY],
                    'class': cls,
                    'conf': conf,
                    'bbox': [x1, y1, x2, y2],
                    'mask': mask
                }

                new.append(current_prediction)
    good_set_1 = tracker.update(new)
    #Returns the bounding boxes which are simmilar to previous detections and get's rid of missed detections


    # Create a copy of the image to draw the mask overlays on
    overlay = frame.copy()
    for prediction in good_set_1:
        # Get the bounding box coordinates and class ID
        box = prediction.bbox
        x1, y1, x2, y2 = box
        class_id = prediction.class_id
        mask = prediction.avg_mask
        mask_2d = mask[0]
        print(prediction.liveness_counter)

        # Determine the color for the class
        num_colors = len(COLORS)
        color = COLORS[int(class_id) % num_colors]

        # Draw the rectangle on the image curently comented to focus on the segmentation

        #cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        overlay[mask_2d == 1] = color
        text = f"Conf: {prediction.conf:.2f}"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1

        # Use cv2.getTextSize to get the width and height of the text box
        (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, font_thickness)
        bg_rect_x1 = x1
        bg_rect_y1 = y1 - text_height - baseline

        # Set the bottom-right corner of the background rectangle
        bg_rect_x2 = x1 + text_width
        bg_rect_y2 = y1

        # Draw the filled rectangle for the class and confidence
        cv.rectangle(frame, (int(bg_rect_x1), int(bg_rect_y1)), (int(bg_rect_x2), int(bg_rect_y2)), color, thickness=cv.FILLED)

        # --- 5. Write the Text on Top of the Background ---
        # Define the text's bottom-left starting position
        text_x = x1
        text_y = y1 - baseline // 2

        # Write the confidence and class
        text_color = (255, 255, 255)  # White in BGR
        cv.putText(frame, text, (int(text_x), int(text_y)), font, font_scale, text_color, font_thickness)

    # --- 5. Display the final image ---
    alpha = 0.6  # Transparency factor for the original image
    beta = 0.4  # Transparency factor for the overlay
    gamma = 0  # Scalar added to each sum
    
    #add the masks
    final_image = cv.addWeighted(frame, alpha, overlay, beta, gamma)
    cv.imshow('final', final_image)
    previous = new
    if cv.waitKey(1) == ord("q"):
        cont = False





