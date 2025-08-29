# This script is a conversion of the speed_estimation_notebook.ipynb file.
# It performs vehicle speed estimation on a video file.

# --- How to Run ---
# 1. Make sure you have all dependencies installed (ultralytics, opencv-python, torch, deep_sort_realtime, numpy).
# 2. Place this script in the same folder as your 'yolov10n.pt' model and the 'configs' folder containing 'coco.names'.
# 3. Run the script from your terminal:
#    python speed_estimation_script.py --video /path/to/your/video.mp4 --output /path/to/save/result.mp4

import cv2
import torch
import time
import numpy as np
import argparse
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

def read_frames(cap):
    """A generator function to read frames from a video capture object."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

def calculate_speed(distance, fps):
    """Calculates speed in km/h."""
    return (distance * fps) * 3.6

def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def draw_corner_rect(img, bbox, line_length=30, line_thickness=5, rect_thickness=1,
                     rect_color=(255, 0, 255), line_color=(0, 255, 0)):
    """Draws a styled bounding box with corner highlights."""
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    if rect_thickness != 0:
        cv2.rectangle(img, bbox, rect_color, rect_thickness)

    # Top Left
    cv2.line(img, (x, y), (x + line_length, y), line_color, line_thickness)
    cv2.line(img, (x, y), (x, y + line_length), line_color, line_thickness)
    # Top Right
    cv2.line(img, (x1, y), (x1 - line_length, y), line_color, line_thickness)
    cv2.line(img, (x1, y), (x1, y + line_length), line_color, line_thickness)
    # Bottom Left
    cv2.line(img, (x, y1), (x + line_length, y1), line_color, line_thickness)
    cv2.line(img, (x, y1), (x, y1 - line_length), line_color, line_thickness)
    # Bottom Right
    cv2.line(img, (x1, y1), (x1 - line_length, y1), line_color, line_thickness)
    cv2.line(img, (x1, y1), (x1, y1 - line_length), line_color, line_thickness)
    return img

def main(args):
    # --- Parameters and Configuration ---
    FRAME_WIDTH = 30
    FRAME_HEIGHT = 100
    SOURCE_POLYGON = np.array([[18, 550], [1852, 608], [1335, 370], [534, 343]], dtype=np.float32)
    BIRD_EYE_VIEW = np.array([[0, 0], [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT], [0, FRAME_HEIGHT]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(SOURCE_POLYGON, BIRD_EYE_VIEW)

    # --- Initialization ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return

    frame_generator = read_frames(cap)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create mask to filter detections within the polygon
    pts = SOURCE_POLYGON.astype(np.int32).reshape((-1, 1, 2))
    polygon_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [pts], 255)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    # Initialize tracker and model
    tracker = DeepSort(max_age=50)
    model = YOLO("yolov10n.pt")
    
    with open("configs/coco.names", "r") as f:
        class_names = f.read().strip().split("\n")
    
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    # --- Main Processing Loop ---
    frame_count = 0
    start_time = time.time()
    prev_positions = {}
    speed_accumulator = {}

    print("Processing video...")
    while True:
        try:
            frame = next(frame_generator)
        except StopIteration:
            break
            
        with torch.no_grad():
            results = model(frame)

        detect = []
        for pred in results:
            for box in pred.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                label = box.cls[0]

                if confidence < args.conf:
                    continue
                
                # Check if the center of the box is within the polygon
                if polygon_mask[(y1 + y2) // 2, (x1 + x2) // 2] == 255:
                    detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, int(label)])

        tracks = tracker.update_tracks(detect, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            
            color = colors[class_id]
            B, G, R = map(int, color)
            text = f"{track_id} - {class_names[class_id]}"

            center_pt = np.array([[(x1 + x2) // 2, (y1 + y2) // 2]], dtype=np.float32)
            transformed_pt = cv2.perspectiveTransform(center_pt[None, :, :], M)

            if track_id in prev_positions:
                distance = calculate_distance(prev_positions[track_id], transformed_pt[0][0])
                speed = calculate_speed(distance, fps)
                speed_accumulator.setdefault(track_id, []).append(speed)
                if len(speed_accumulator[track_id]) > 100:
                    speed_accumulator[track_id].pop(0)
            
            prev_positions[track_id] = transformed_pt[0][0]

            frame = draw_corner_rect(frame, (x1, y1, x2 - x1, y2 - y1), line_length=15, line_thickness=3)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 10, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if track_id in speed_accumulator:
                avg_speed = sum(speed_accumulator[track_id]) / len(speed_accumulator[track_id])
                color_box = (0, 255, 0) if avg_speed <= args.speed_limit else (0, 0, 255)
                cv2.rectangle(frame, (x1 - 1, y1 - 40), (x1 + 150, y1 - 20), color_box, -1)
                cv2.putText(frame, f"Speed: {avg_speed:.0f} km/h", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        writer.write(frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {frame_count} frames. FPS: {frame_count / elapsed_time:.2f}")

    # --- Cleanup ---
    cap.release()
    writer.release()
    print(f"Processing complete. Video saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vehicle Speed Estimation Script")
    parser.add_argument('--video', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output video file.")
    parser.add_argument('--speed_limit', type=int, default=60, help="Speed limit in km/h for highlighting.")
    parser.add_argument('--conf', type=float, default=0.5, help="Confidence threshold for object detection.")
    args = parser.parse_args()
    main(args)