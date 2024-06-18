import cv2
import os
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from some_yolo_library import YOLO  # Replace with actual YOLO library import

# Initialize YOLO and DeepSort
damage_yolo = YOLO("damage_model_config.cfg", "damage_model_weights.weights")
car_yolo = YOLO("car_model_config.cfg", "car_model_weights.weights")
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# Define class indices you want to track
damage_class = [0]  # Example: class index for damages
car_class = [1]  # Example: class index for car

# Variables to store extracted damages
extracted_damages = {}
frame_index = 0

# Open video capture
cap = cv2.VideoCapture("input_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect damages and car segments
    damage_detections = []
    car_detections = []

    damage_results = damage_yolo.detect(frame)
    car_results = car_yolo.detect(frame)

    for (x1, y1, x2, y2, conf, cls) in damage_results:
        if conf > 0.2 and cls in damage_class:
            bbox = [x1, y1, x2 - x1, y2 - y1]
            damage_detections.append((bbox, conf, int(cls)))

    for (x1, y1, x2, y2, conf, cls) in car_results:
        if conf > 0.2 and cls in car_class:
            bbox = [x1, y1, x2 - x1, y2 - y1]
            car_detections.append((bbox, conf, int(cls)))

    # Update tracker with damage detections
    tracked_objects = tracker.update_tracks(damage_detections, frame=frame)

    # Process tracked objects and calculate IoU with car detections
    for track in tracked_objects:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        damage_bbox = track.to_tlwh(orig=True)
        damage_area = damage_bbox[2] * damage_bbox[3]

        max_iou = 0
        for car_bbox, _, _ in car_detections:
            car_bbox = [int(i) for i in car_bbox]
            iou = calculate_iou(damage_bbox, car_bbox)
            max_iou = max(max_iou, iou)

        # Save frame if it has significant overlap
        if track_id not in extracted_damages or extracted_damages[track_id]['max_iou'] < max_iou:
            extracted_damages[track_id] = {
                'frame_index': frame_index,
                'bbox': damage_bbox,
                'image': frame,
                'max_iou': max_iou
            }

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# Select key frames per track
final_frames = []
for track_id, data in extracted_damages.items():
    final_frames.append(data)

# Sort frames by max_iou and select top frames
final_frames = sorted(final_frames, key=lambda x: x['max_iou'], reverse=True)[:15]

# Save the extracted frames
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
for i, frame_data in enumerate(final_frames):
    cv2.imwrite(os.path.join(output_dir, f"frame_{i}.jpg"), frame_data['image'])

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou
