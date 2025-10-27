import cv2
import numpy as np
#from ultralytics import solutions
from ultralytics.utils.downloads import safe_download
from ultralytics import YOLO
from collections import defaultdict
import yaml
import tkinter as tk
from tkinter import filedialog
import os
# -------------------------------
# Step 1: Choose Input Source
# -------------------------------

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

# Build full path to parms.yaml
yaml_path = os.path.join(BASE_DIR, "parms.yaml")
print(yaml_path)
with open(yaml_path) as f:
    params = yaml.safe_load(f)

print("Select Input Source:")
print("1. Local video file")
print("2. RTSP / Webcam stream")
choice = input("Enter 1 or 2: ")

if choice == "1":
        # Open file chooser dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    video_source = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )

    if not video_source:
        print("No file selected. Exiting...")
        exit()
else:
    video_source = input("Enter RTSP URL or webcam index (e.g. 0): ")
    try:
        video_source = int(video_source)
    except:
        pass  # Keep as RTSP string if not int

cap = cv2.VideoCapture(video_source)
assert cap.isOpened(), "Error: Unable to open video source"

# -------------------------------
# Step 2: Select Region Points
# -------------------------------
print("\nINSTRUCTION:")
print(" Select 2 points for LINE counting, or 4 points for RECTANGLE region counting.")
print(" Left-click to select each point. Press ENTER when done.")

region_points = []
in_count, out_count = 0, 0
frame_counter = 0
prev_positions = {}  # track each object’s previous position
drawing_window = "Select Region"
cv2.namedWindow(drawing_window)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        region_points.append((x, y))
        print(f"Point selected: ({x}, {y})")

def get_side_of_line(p1, p2, cx, cy):
    # returns positive or negative depending on which side of the line (cx, cy) is
    return (p2[0] - p1[0]) * (cy - p1[1]) - (p2[1] - p1[1]) * (cx - p1[0])

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Error reading initial frame for region selection.")

clone = frame.copy()
cv2.setMouseCallback(drawing_window, mouse_callback)

while True:
    temp_frame = clone.copy()
    # Draw current points
    for p in region_points:
        cv2.circle(temp_frame, p, 5, (0, 255, 0), -1)

    # Draw line or polygon
    if len(region_points) >= 2:
        cv2.polylines(temp_frame, [np.array(region_points, np.int32)], False, (255, 0, 0), 2)

    cv2.imshow(drawing_window, temp_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # Enter key
        break
    elif key == 27:  # ESC key to quit
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow(drawing_window)

if len(region_points) not in [2, 4]:
    raise ValueError("You must select 2 points for a line or 4 points for a rectangle.")

print(f"\n Region points selected: {region_points}")

# -------------------------------
# Step 3: Initialise Video Writer
# -------------------------------
w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS
))
video_writer = cv2.VideoWriter(
    "counting_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps if fps > 0 else 30,
    (w, h)
)

# -------------------------------
# Step 4:Helper functions for results visualisation 
# -------------------------------
def draw_neon_corner_box(frame, x1, y1, x2, y2, color=(0, 255, 255), thickness=2, corner_len=15, glow_intensity=0.4):
    """
    Draws a glowing neon-style corner box around the object.
    Combines neon glow + corner-only minimalistic box.
    """

    # --- Neon glow overlay ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, glow_intensity, frame, 1 - glow_intensity, 0, frame)

    # --- Corner-style edges ---
    # top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
    # top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
    # bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
    # bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    return frame

# -------------------------------
# Step 5: Process the Video
# -------------------------------
print("\n Starting object counting... Press 'q' to quit early.\n")

model_path=params['Inference_model_path']
model = YOLO(model_path)
class_list = model.names
# Object counting
crossed_ids = set()               # permanently crossed IDs
just_crossed_ids = {}             # temporarily flash yellow
class_counts_in = defaultdict(int)
class_counts_out = defaultdict(int)
prev_sides = {}
conf_thresh = 0.7
# Number of frames to keep the yellow flash
FLASH_FRAMES = 4

frame_num = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    # Run YOLO inference with confidence filter
 
    results = model.track(frame, persist=True, classes=params['classes_to_track'], conf=conf_thresh) 

    if not results or results[0].boxes is None:
        cv2.imshow("YOLO Object Tracking & Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    try:

        boxes = results[0].boxes.xyxy.cpu()
        confs = results[0].boxes.conf.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()

    except Exception as e:
        print(f"Error processing frame {frame_num}: {e}")
        continue

    # Draw the user-defined line
    if len(region_points) == 2:
        cv2.line(frame, region_points[0], region_points[1], (0, 0, 255), 3)

    for box, conf, track_id, class_idx in zip(boxes, confs, track_ids, class_indices):
        if conf < conf_thresh:
            continue  # skip low-confidence detections

        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        class_name = class_list[class_idx]

        # Default color: red
        color = (0, 0, 255)

        # Yellow flash after crossing
        if track_id in just_crossed_ids:
            if frame_num - just_crossed_ids[track_id] < FLASH_FRAMES:
                color = (0, 255, 255)
            else:
                crossed_ids.add(track_id)
                del just_crossed_ids[track_id]
        elif track_id in crossed_ids:
            color = (0, 255, 0)

        # Draw bbox and center
        
        draw_neon_corner_box(frame, x1, y1, x2, y2, color)
        #cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Skip counting if line not set
        if len(region_points) != 2:
            continue

        # Crossing detection
        p1, p2 = region_points
        current_side = get_side_of_line(p1, p2, cx, cy)
        if track_id not in prev_sides:
            prev_sides[track_id] = current_side
            continue

        prev_side = prev_sides[track_id]
        if prev_side * current_side < 0:  # crossed line
            if prev_side < 0 and current_side > 0 and track_id not in crossed_ids:
                class_counts_out[class_name] += 1
                just_crossed_ids[track_id] = frame_num
            elif prev_side > 0 and current_side < 0 and track_id not in crossed_ids:
                class_counts_in[class_name] += 1
                just_crossed_ids[track_id] = frame_num

        prev_sides[track_id] = current_side

    # Show counts
    y_offset = 30
    for cls in sorted(set(list(class_counts_in.keys()) + list(class_counts_out.keys()))):
        cv2.putText(frame, f"{cls} IN: {class_counts_in[cls]}  OUT: {class_counts_out[cls]}",
                    (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30
    video_writer.write(frame)
    cv2.imshow("YOLO Object Tracking & Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("\n Counting complete! Output saved as 'counting_output.mp4'")
