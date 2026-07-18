"""
Human Pose Estimation & Action Recognition in Crowded Scenes
Streamlit app for Streamlit Community Cloud.

Same pipeline as your Colab notebook (Blocks 2-8), wrapped for Streamlit.
"""

import cv2
import math
import subprocess
import tempfile
import os
import streamlit as st
from ultralytics import YOLO

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Crowd Pose & Action Recognition", layout="centered")

# ==========================================
# LOAD MODEL (cached so it only loads once, not on every interaction)
# ==========================================
@st.cache_resource
def load_model():
    return YOLO("yolov8n-pose.pt")

model = load_model()

# ==========================================
# COLOR PALETTE (same as your notebook)
# ==========================================
SKELETON_COLOR = (255, 255, 0)   # Cyan (BGR)
JOINT_COLOR = (255, 255, 255)    # White
LABEL_BG = (0, 0, 0)             # Black

ACTION_COLORS = {
    "Standing": (0, 255, 0),
    "Walking": (255, 0, 0),
    "Sitting": (0, 255, 255),
    "Hand Raised": (255, 0, 255),
    "Bending": (0, 165, 255),
    "Unknown": (200, 200, 200)
}

SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


# ==========================================
# DRAWING FUNCTIONS
# ==========================================
def draw_skeleton(frame, keypoints):
    for p1, p2 in SKELETON:
        if keypoints[p1][2] > 0.5 and keypoints[p2][2] > 0.5:
            pt1 = (int(keypoints[p1][0]), int(keypoints[p1][1]))
            pt2 = (int(keypoints[p2][0]), int(keypoints[p2][1]))
            cv2.line(frame, pt1, pt2, SKELETON_COLOR, 3)

    for kp in keypoints:
        x, y, conf = kp
        if conf > 0.5:
            cv2.circle(frame, (int(x), int(y)), 4, JOINT_COLOR, -1)


def place_label_no_overlap(frame, text, x, y, color, placed_boxes):
    font = cv2.FONT_HERSHEY_COMPLEX
    scale = 0.55
    thickness = 2

    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    box_x1, box_y1 = x, y - h - 12
    box_x2, box_y2 = x + w + 12, y + 5

    def overlaps(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

    shift = 0
    max_shifts = 6
    while any(overlaps((box_x1, box_y1 + shift, box_x2, box_y2 + shift), pb) for pb in placed_boxes) \
            and shift < max_shifts * (h + 8):
        shift += (h + 8)

    box_y1 += shift
    box_y2 += shift
    text_y = y + shift

    fh, fw = frame.shape[:2]
    if box_x2 > fw:
        dx = box_x2 - fw
        box_x1 -= dx
        box_x2 -= dx
    if box_y1 < 0:
        box_y1 = 2
        box_y2 = box_y1 + h + 17
        text_y = box_y2 - 5

    cv2.rectangle(frame, (int(box_x1), int(box_y1)), (int(box_x2), int(box_y2)), LABEL_BG, -1)
    cv2.putText(frame, text, (int(box_x1) + 5, int(text_y) - 5), font, scale, color, thickness)

    placed_boxes.append((box_x1, box_y1, box_x2, box_y2))
    return placed_boxes


# ==========================================
# ACTION RECOGNITION
# ==========================================
def recognize_action(keypoints, track_id, previous_positions):
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    nose = keypoints[0]

    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_y = (left_hip[1] + right_hip[1]) / 2
    knee_y = (left_knee[1] + right_knee[1]) / 2

    center_x = (left_hip[0] + right_hip[0]) / 2
    center_y = hip_y

    walking = False
    if track_id in previous_positions:
        prev_x, prev_y = previous_positions[track_id]
        movement = math.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
        if movement > 8:
            walking = True
    previous_positions[track_id] = (center_x, center_y)

    if left_wrist[1] < shoulder_y or right_wrist[1] < shoulder_y:
        return "Hand Raised"
    if abs(hip_y - knee_y) < 45:
        return "Sitting"
    if nose[1] > shoulder_y:
        return "Bending"
    if walking:
        return "Walking"
    return "Standing"


# ==========================================
# MAIN PIPELINE
# ==========================================
def process_video(input_path, output_path, progress_bar, status_text):
    cap = cv2.VideoCapture(input_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    previous_positions = {}
    last_action = {}
    unique_ids_seen = set()
    action_count = {"Standing": 0, "Walking": 0, "Sitting": 0,
                     "Hand Raised": 0, "Bending": 0, "Unknown": 0}

    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))
        status_text.text(f"Processing frame {frame_count}/{total_frames}")

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        r = results[0]

        placed_boxes = []
        people_in_frame = 0

        if r.keypoints is not None and r.boxes is not None and r.boxes.id is not None:
            keypoints_all = r.keypoints.data.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            people_in_frame = len(track_ids)

            for kp, tid in zip(keypoints_all, track_ids):
                unique_ids_seen.add(tid)
                draw_skeleton(frame, kp)

                action = recognize_action(kp, tid, previous_positions)
                action = last_action.get(tid, action) if action == "Unknown" else action
                last_action[tid] = action
                action_count[action] = action_count.get(action, 0) + 1

                color = ACTION_COLORS.get(action, ACTION_COLORS["Unknown"])

                nose = kp[0]
                if nose[2] > 0.3:
                    label_x, label_y = int(nose[0]) - 10, int(nose[1]) - 25
                else:
                    ls, rs = kp[5], kp[6]
                    label_x = int((ls[0] + rs[0]) / 2) - 10
                    label_y = int(min(ls[1], rs[1])) - 25

                label_x = max(0, label_x)
                label_y = max(20, label_y)

                label_text = f"ID {tid}: {action}"
                placed_boxes = place_label_no_overlap(frame, label_text, label_x, label_y, color, placed_boxes)

        count_text = f"People in Frame: {people_in_frame}"
        (cw, ch), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)
        cv2.rectangle(frame, (15, 15), (25 + cw, 45 + ch), LABEL_BG, -1)
        cv2.putText(frame, count_text, (20, 40 + ch - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    return unique_ids_seen, action_count


# ==========================================
# STREAMLIT UI
# ==========================================
st.title("Human Pose Estimation & Action Recognition in Crowded Scenes")
st.write(
    "Upload a video. The model detects people, tracks them with stable IDs, "
    "draws cyan skeletons with white joints, and labels each person's action "
    "(Standing, Sitting, Walking, Hand Raised, Bending)."
)

uploaded_file = st.file_uploader("Upload a crowd video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    if st.button("Run Pose & Action Recognition"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "input.mp4")
            raw_output_path = os.path.join(tmp_dir, "output_raw.mp4")
            playable_output_path = os.path.join(tmp_dir, "output_playable.mp4")

            with open(input_path, "wb") as f:
                f.write(uploaded_file.read())

            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Processing video..."):
                unique_ids_seen, action_count = process_video(
                    input_path, raw_output_path, progress_bar, status_text
                )

                # Re-encode to H.264 so it plays in the browser
                subprocess.run(
                    ["ffmpeg", "-y", "-i", raw_output_path, "-vcodec", "libx264",
                     "-pix_fmt", "yuv420p", playable_output_path],
                    check=True
                )

            status_text.text("Done!")
            st.success(f"Unique people detected: {len(unique_ids_seen)}")
            st.json(action_count)

            st.video(playable_output_path)

            with open(playable_output_path, "rb") as f:
                st.download_button(
                    "Download processed video",
                    f,
                    file_name="output_action_recognition.mp4",
                    mime="video/mp4"
                )
