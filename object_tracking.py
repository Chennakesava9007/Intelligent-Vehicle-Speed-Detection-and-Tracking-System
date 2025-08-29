import cv2
import torch
import numpy as np
import argparse, os
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from ultralytics import YOLO

# Unified Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, required=True, help="Path to input video")
parser.add_argument('--output', type=str, required=True, help="Path to save output video")
parser.add_argument('--speed_limit', type=int, default=60, help="Speed limit in km/h")
parser.add_argument('--conf', type=float, default=0.50, help="Confidence threshold")
parser.add_argument('--blur_id', type=int, default=None, help="Class ID to apply Gaussian Blur")
parser.add_argument('--class_id', type=int, default=None, help="Class ID to track")
args = parser.parse_args()

VIDEO_PATH = args.video
OUTPUT_PATH = args.output
SPEED_LIMIT = args.speed_limit


def draw_corner_rect(img, bbox, line_length=30, line_thickness=5, rect_thickness=1,
                     rect_color=(255, 0, 255), line_color=(0, 255, 0)):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    if rect_thickness != 0:
        cv2.rectangle(img, bbox, rect_color, rect_thickness)

    cv2.line(img, (x, y), (x + line_length, y), line_color, line_thickness)
    cv2.line(img, (x, y), (x, y + line_length), line_color, line_thickness)
    cv2.line(img, (x1, y), (x1 - line_length, y), line_color, line_thickness)
    cv2.line(img, (x1, y), (x1, y + line_length), line_color, line_thickness)
    cv2.line(img, (x, y1), (x + line_length, y1), line_color, line_thickness)
    cv2.line(img, (x, y1), (x, y1 - line_length), line_color, line_thickness)
    cv2.line(img, (x1, y1), (x1 - line_length, y1), line_color, line_thickness)
    cv2.line(img, (x1, y1), (x1, y1 - line_length), line_color, line_thickness)
    return img

def calculate_speed(distance, fps):
    return (distance * fps) * 3.6

def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def read_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

FRAME_WIDTH = 30
FRAME_HEIGHT = 100
SOURCE_POLYGONE = np.array([[18, 550], [1852, 608], [1335, 370], [534, 343]], dtype=np.float32)
BIRD_EYE_VIEW = np.array([[0, 0], [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT], [0, FRAME_HEIGHT]], dtype=np.float32)
M = cv2.getPerspectiveTransform(SOURCE_POLYGONE, BIRD_EYE_VIEW)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print('Error: Unable to open video source.')
    exit()

frame_generator = read_frames(cap)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

pts = SOURCE_POLYGONE.astype(np.int32).reshape((-1, 1, 2))
polygon_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv2.fillPoly(polygon_mask, [pts], 255)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

tracker = DeepSort(max_age=50)
model = YOLO("yolov10n.pt")

with open("configs/coco.names", "r") as f:
    class_names = f.read().strip().split("\n")

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3))

frame_count = 0
start_time = time.time()
prev_positions = {}
speed_accumulator = {}

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
            if args.class_id is not None and label != args.class_id:
                continue
            if confidence < args.conf:
                continue
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
        if polygon_mask[(y1 + y2) // 2, (x1 + x2) // 2] == 0:
            continue

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

        frame = draw_corner_rect(frame, (x1, y1, x2 - x1, y2 - y1), line_length=15, line_thickness=3,
                                 rect_thickness=1, rect_color=(B, G, R), line_color=(R, G, B))
        cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 10, y1), (B, G, R), -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if track_id in speed_accumulator:
            avg_speed = sum(speed_accumulator[track_id]) / len(speed_accumulator[track_id])
            color_box = (0, 255, 0) if avg_speed <= SPEED_LIMIT else (0, 0, 255)
            cv2.rectangle(frame, (x1 - 1, y1 - 40), (x1 + len(f"Speed: {avg_speed:.0f} km/h") * 10, y1 - 20),
                          color_box, -1)
            cv2.putText(frame, f"Speed: {avg_speed:.0f} km/h", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)

        if args.blur_id is not None and class_id == args.blur_id:
            if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

    cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.imshow('speed_estimation', frame)
    writer.write(frame)
    frame_count += 1
    if frame_count % 10 == 0:
        elapsed_time = time.time() - start_time
        print(f"FPS: {frame_count / elapsed_time:.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()