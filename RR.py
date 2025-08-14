#This file can only be run if the correct model is in the same file as the rest please download the YOLO STFA model



import math
import random

import cv2 as cv
from ultralytics import YOLO
import numpy as np
from Tracker import Tracker
from PyQt5.QtCore import  QThread, pyqtSignal



# --- Paste the content of your Tracker.py file here ---


# --- Worker thread for video processing ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_table_signal = pyqtSignal(list)

    def __init__(self, camera_index):
        super().__init__()
        self._run_flag = True
        self.conf = 0.5
        self.iou = 0.1
        self.distance_threshold = 25.0
        self.smoothing_factor = 0.5
        self.max_liveliness = 25.0
        self.model = YOLO('YOLOSTFA.pt')
        self.cap = cv.VideoCapture(camera_index, cv.CAP_DSHOW)

    def run(self):

        tracker = Tracker(distance_threshold=self.distance_threshold, smoothing_factor=self.smoothing_factor,
                          max_liveliness=self.max_liveliness)
        memory = []
        framesused = []
        COLORS = [
            (255, 0, 0),  # Blue
            (0, 255, 0),  # Green
            (0, 0, 255),  # Red
            (255, 255, 0),  # Teal
        ]

        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret:
                continue

            tracker.update_params(self.distance_threshold, self.smoothing_factor, self.max_liveliness)

            if len(memory) < 30:
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
            concatenated_array = np.concatenate([img3_arr, img2_arr, img1_arr], axis=2)

            results = self.model(concatenated_array, conf=self.conf, iou=self.iou, device=0)

            new_predictions = []
            if results and results[0].masks is not None and results[0].boxes is not None:
                for mask, box in zip(results[0].masks, results[0].boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(np.int32)
                    middleX = math.ceil(x1 + (x2 - x1) / 2)
                    middleY = math.ceil(y1 + (y2 - y1) / 2)
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    mask_data = mask.data.cpu().numpy()
                    current_prediction = {
                        'middle': [middleX, middleY],
                        'class': cls,
                        'conf': conf,
                        'bbox': [x1, y1, x2, y2],
                        'mask': mask_data
                    }
                    new_predictions.append(current_prediction)

            good_set = tracker.update(new_predictions)
            self.update_table_signal.emit(good_set)

            overlay = frame.copy()
            for prediction in good_set:
                # FIX 2: Add a defensive check to ensure the mask is valid before trying to use it for drawing.
                # This prevents an IndexError or TypeError if the mask data is somehow corrupted or missing.
                mask = prediction.avg_mask
                if mask is None or mask.ndim < 2 or mask.shape[0] == 0:
                    continue

                box = prediction.bbox
                x1, y1, x2, y2 = map(int, box)
                class_id = prediction.class_id
                mask_2d = mask[0]

                color = COLORS[int(class_id) % len(COLORS)]

                overlay[mask_2d == 1] = color
                text = f"Conf: {prediction.conf:.2f}"
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 1

                (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, font_thickness)
                bg_rect_x1, bg_rect_y1 = x1, y1 - text_height - baseline
                bg_rect_x2, bg_rect_y2 = x1 + text_width, y1

                cv.rectangle(frame, (bg_rect_x1, bg_rect_y1), (bg_rect_x2, bg_rect_y2), color, -1)
                cv.putText(frame, text, (x1, y1 - baseline // 2), font, font_scale, (255, 255, 255), font_thickness)

            final_image = cv.addWeighted(frame, 0.6, overlay, 0.4, 0)
            self.change_pixmap_signal.emit(final_image)

        self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def update_params(self, conf, iou, distance_threshold, smoothing_factor, max_liveliness):
        self.conf = conf
        self.iou = iou
        self.distance_threshold = distance_threshold
        self.smoothing_factor = smoothing_factor
        self.max_liveliness = max_liveliness






