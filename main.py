import cv2
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from some_yolo_library import YOLO  # Replace with actual YOLO library import

# Initialize YOLO and DeepSort
yolo = YOLO("model_config.cfg", "model_weights.weights")
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# Define class indices you want to track
classes = [0]  # Example: class index for damages

# Variables to store extracted damages
extracted_damages = {}

# Open video capture
cap = cv2.VideoCapture("input_video.mp4")
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    detections = []
    results = yolo.detect(frame)
    for (x1, y1, x2, y2, conf, cls) in results:
        if conf > 0.2 and cls in classes:  # Filter by class index
            bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [left, top, width, height]
            detections.append((bbox, conf, int(cls)))

    # Update tracker
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    # Process tracked objects
    for track in tracked_objects:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        bbox = track.to_tlwh(orig=True)  # Use original bounding box
        x, y, w, h = map(int, bbox)

        # Save frames ensuring the entire car is visible and the damage is most prominent
        if track_id not in extracted_damages:
            extracted_damages[track_id] = []
        extracted_damages[track_id].append({
            'frame_index': frame_index,
            'bbox': bbox,
            'image': frame
        })
    
    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# Select key frames per track
final_frames = {}
for track_id, frames in extracted_damages.items():
    selected_frames = []
    frames = sorted(frames, key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)  # Sort by area
    selected_frames = frames[:min(15, len(frames))]  # Select top 15 or fewer frames
    final_frames[track_id] = selected_frames

# Save the extracted frames
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
for track_id, frames in final_frames.items():
    for i, frame_data in enumerate(frames):
        cv2.imwrite(os.path.join(output_dir, f"{track_id}_{i}.jpg"), frame_data['image'])
