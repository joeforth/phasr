from ultralytics import YOLO
import cv2

# model = YOLO("YOLOV11modelv9.pt")  # initialize model
model = YOLO("AOTOBJECTDETECTIONYOLO.pt")  # initialize model

def cluster_x_values(x_values, threshold=30):
    if not x_values:
        return []
    # Sort data by x value
    x_values.sort(key=lambda tup: tup[1])

    # Grouping
    groups = []
    current_group = [x_values[0]]

    for current in x_values[1:]:
        if abs(current[1] - current_group[-1][1]) <= threshold:
            current_group.append(current)
        else:
            groups.append(current_group)
            current_group = [current]
    groups.append(current_group)
    return groups


def getSize(vial, vial_size):
    # Given bounding box width in pixels for the 6 cm reference object
    size_cm = []
    height = next((height for id_val, x, y, height in vial if id_val == 0), None)

    # Check if a reference object was found
    if height is None:
        raise ValueError("Reference bounding box (id == 0) not found in vial data.")

    # Calculate scale factor (Pixels per cm)
    scale_factor = height / vial_size  # Pixels per cm
    for i in vial:
        if i[0] > 0.5:
            height = i[3]
            other_bbox_width_cm = height / scale_factor
            size_cm.append((i[0], i[1], i[2], other_bbox_width_cm))
    return size_cm


def ProcessImage(image, vial_size):
    x_arr = []
    results = model.predict(source=image, conf=0.65, iou=0.3, show_labels=False,device="cpu")

    # Extract bounding boxes
    img = results[0].plot(labels=False, conf=False)  # returns a numpy array

    for result in results:
        for box in result.boxes:
            class_id = box.cls[0].item()  # Class label
            boxes = box.xywh  # Alternative format

            for boxe in boxes:
                x, y, w, h = boxe
                x_arr.append((class_id, x.item(), y.item(), h.item()))

    vials = cluster_x_values(x_arr)
    finalVials = []
    #measure the phase size and display it
    for i in range(len(vials)):
        try:
            size = getSize(vials[i], vial_size[i])
            finalVials.append(size)
            for j in range(len(size)):
                text = f"{size[j][3]:.2f} cm"
                position = (int(size[j][1]), int(size[j][2]))  # (x, y) coordinates
                # Define font, font scale, color (BGR), and thickness
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (0, 255, 255)
                thickness = 1
                cv2.putText(img, text, position, font, font_scale, color, thickness)
        except ValueError as e:
            print(f"Skipping group — no reference found: {e}")
            continue
        except IndexError:
            # No provided size for this vial — skip it
            continue

    return img, len(vials), finalVials
