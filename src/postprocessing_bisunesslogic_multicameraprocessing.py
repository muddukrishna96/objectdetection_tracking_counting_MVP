import cv2
import threading
from collections import defaultdict
from ultralytics import YOLO
import yaml
import os
# -------------------------------
# Step 1: User selects cameras
# -------------------------------
# Get directory where this script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

# Build full path to parms.yaml
yaml_path = os.path.join(BASE_DIR, "parms.yaml")
print(yaml_path)
with open(yaml_path) as f:
    params = yaml.safe_load(f)


print("Enter camera indices (max 2, separated by space): e.g., 0 1")
camera_indices = input("Camera indices: ").split()
camera_indices = [int(i) for i in camera_indices[:2]]

if not camera_indices:
    raise ValueError("No camera indices provided!")

print(f"Selected cameras: {camera_indices}")

# -------------------------------
# Step 2: Shared YOLO Model
# -------------------------------
model_path = params['Inference_model_path']
model = YOLO(model_path)
class_list = model.names
conf_thresh = 0.7

# -------------------------------
# Step 3: Helper functions
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

def get_side_of_line(p1, p2, cx, cy):
    return (p2[0] - p1[0]) * (cy - p1[1]) - (p2[1] - p1[1]) * (cx - p1[0])

# -------------------------------
# Step 4: Let user draw line per camera
# -------------------------------
def select_line_for_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    assert cap.isOpened(), f"Cannot open camera {camera_index}"

    print(f"\nCamera {camera_index}: Select 2 points for line counting. Press ENTER when done.")
    region_points = []
    window_name = f"Draw Line - Camera {camera_index}"
    cv2.namedWindow(window_name)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            region_points.append((x, y))
            print(f"Camera {camera_index}: Point selected {x, y}")

    ret, frame = cap.read()
    clone = frame.copy()
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp = clone.copy()
        for p in region_points:
            cv2.circle(temp, p, 5, (0, 255, 0), -1)
        if len(region_points) >= 2:
            cv2.line(temp, region_points[0], region_points[1], (255, 0, 0), 2)
        cv2.imshow(window_name, temp)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            break
        elif key == 27:  # Esc
            cap.release()
            cv2.destroyWindow(window_name)
            exit()

    cap.release()
    cv2.destroyWindow(window_name)
    return region_points

camera_lines = []
for cam in camera_indices:
    line = select_line_for_camera(cam)
    if len(line) != 2:
        raise ValueError("You must select exactly 2 points for line counting!")
    camera_lines.append(line)

# -------------------------------
# Step 5: Thread function per camera
# -------------------------------
def process_camera(camera_index, region_points, model):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error opening camera {camera_index}")
        return

    print(f"Starting object counting on Camera {camera_index}...")

    crossed_ids = set()
    just_crossed_ids = {}
    class_counts_in = defaultdict(int)
    class_counts_out = defaultdict(int)
    prev_sides = {}
    frame_num = 0
    FLASH_FRAMES = 4

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        results = model.track(frame, persist=True, classes=params['classes_to_track'], conf=conf_thresh)
        if not results or results[0].boxes is None:
            cv2.imshow(f"Camera {camera_index}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        try:
            boxes = results[0].boxes.xyxy.cpu()
            confs = results[0].boxes.conf.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_indices = results[0].boxes.cls.int().cpu().tolist()
        except Exception as e:
            print(f"[Camera {camera_index}] Error processing frame: {e}")
            continue

        p1, p2 = region_points
        cv2.line(frame, p1, p2, (0, 0, 255), 3)

        for box, conf, track_id, class_idx in zip(boxes, confs, track_ids, class_indices):
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            class_name = model.names[class_idx]
            color = (0, 0, 255)
            if track_id in just_crossed_ids:
                if frame_num - just_crossed_ids[track_id] < FLASH_FRAMES:
                    color = (0, 255, 255)
                else:
                    crossed_ids.add(track_id)
                    del just_crossed_ids[track_id]
            elif track_id in crossed_ids:
                color = (0, 255, 0)

            draw_neon_corner_box(frame, x1, y1, x2, y2, color)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            current_side = get_side_of_line(p1, p2, cx, cy)
            if track_id not in prev_sides:
                prev_sides[track_id] = current_side
                continue
            prev_side = prev_sides[track_id]
            if prev_side * current_side < 0:
                if prev_side < 0 and current_side > 0 and track_id not in crossed_ids:
                    class_counts_out[class_name] += 1
                    just_crossed_ids[track_id] = frame_num
                elif prev_side > 0 and current_side < 0 and track_id not in crossed_ids:
                    class_counts_in[class_name] += 1
                    just_crossed_ids[track_id] = frame_num
            prev_sides[track_id] = current_side

        y_offset = 30
        for cls in sorted(set(list(class_counts_in.keys()) + list(class_counts_out.keys()))):
            cv2.putText(frame, f"{cls} IN: {class_counts_in[cls]}  OUT: {class_counts_out[cls]}",
                        (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 30

        cv2.imshow(f"Camera {camera_index}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(f"Camera {camera_index}")
    print(f"Camera {camera_index} processing stopped.")

# -------------------------------
# Step 6: Start threads
# -------------------------------
threads = []
for i, cam in enumerate(camera_indices):
    t = threading.Thread(target=process_camera, args=(cam, camera_lines[i], model))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

cv2.destroyAllWindows()
print("\nAll cameras stopped. Exiting.")
