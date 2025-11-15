import cv2
import mediapipe as mp
import time

# ---------------- Mediapipe Face Mesh Setup ----------------
# NOTE: This must be initialized and closed in the final implementation.
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Landmark indices (UNCHANGED)
NOSE_TIP = 1
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
FOREHEAD = 10
CHIN = 152

# Combined Calibration and Smoothing Parameters (UNCHANGED)
CALIBRATION_OFFSET_X = 0.05
SMOOTHING_FACTOR_ALPHA = 0.3
SENSITIVITY = 18.0

# Smoothing variables must be global or passed around if this logic is used outside a class
smoothed_x = 0.5
smoothed_y = 0.5


def get_iris_center(landmarks, iris_indices):
    """Calculates the center point (x, y) of the iris based on its four corner landmarks."""
    x = sum(landmarks[i].x for i in iris_indices) / len(iris_indices)
    y = sum(landmarks[i].y for i in iris_indices) / len(iris_indices)
    return x, y


def calculate_combined_gaze(landmarks):
    """
    Calculates the combined, smoothed gaze vector (smoothed_x, smoothed_y) 
    normalized between 0.0 and 1.0.
    """
    global smoothed_x, smoothed_y # Must be handled by caller context

    # Shared Landmarks
    nose = landmarks[NOSE_TIP]
    left_iris_x, left_iris_y = get_iris_center(landmarks, LEFT_IRIS)
    right_iris_x, right_iris_y = get_iris_center(landmarks, RIGHT_IRIS)

    # ---------------- 1. X-AXIS (Horizontal, Eye-Dominant Logic) ----------------
    left_eye_outer = landmarks[LEFT_EYE_OUTER]
    right_eye_outer = landmarks[RIGHT_EYE_OUTER]
    
    face_width = abs(left_eye_outer.x - right_eye_outer.x)
    
    # Calculate key centers
    left_eye_center = (landmarks[LEFT_EYE_INNER].x + landmarks[LEFT_EYE_OUTER].x) / 2
    right_eye_center = (landmarks[RIGHT_EYE_INNER].x + landmarks[RIGHT_EYE_OUTER].x) / 2
    face_center_x = (landmarks[LEFT_EYE_INNER].x + landmarks[RIGHT_EYE_INNER].x) / 2

    # Calculate offsets
    left_offset_x = (left_iris_x - left_eye_center) / face_width
    right_offset_x = (right_iris_x - right_eye_center) / face_width
    eye_gaze_x = (left_offset_x + right_offset_x) / 2
    head_offset_x = (nose.x - face_center_x) / face_width

    # X-Fusion: W_E=0.75, W_H=0.25 
    combined_x_raw = 0.5 + eye_gaze_x * 0.75 + head_offset_x * 0.25
    combined_x = 0.5 + (combined_x_raw - 0.5) * SENSITIVITY - CALIBRATION_OFFSET_X
    combined_x = max(0, min(1, combined_x))

    # ---------------- 2. Y-AXIS (Vertical, Head-Dominant Logic) ----------------
    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]
    
    face_height = abs(forehead.y - chin.y)
    face_center_y = (forehead.y + chin.y) / 2
    
    iris_y = (left_iris_y + right_iris_y) / 2
    
    eye_gaze_y = (iris_y - face_center_y) / face_height 
    head_offset_y = (nose.y - face_center_y) / face_height 
    
    # Y-Fusion: W_E=0.3, W_H=0.7 
    combined_y_raw = 0.5 + (eye_gaze_y * 0.3) + (head_offset_y * 0.7)
    
    combined_y = 0.5 + (combined_y_raw - 0.5) * SENSITIVITY
    combined_y = max(0, min(1, combined_y))

    # ---------------- 3. Smoothing ----------------
    smoothed_x = SMOOTHING_FACTOR_ALPHA * combined_x + (1 - SMOOTHING_FACTOR_ALPHA) * smoothed_x
    smoothed_y = SMOOTHING_FACTOR_ALPHA * combined_y + (1 - SMOOTHING_FACTOR_ALPHA) * smoothed_y

    return smoothed_x, smoothed_y


def map_gaze_to_card(gaze_x, gaze_y):
    """Maps (gaze_x, gaze_y) to one of the 4 cards in a 2-row, 2-column grid (index 0-3)."""
    
    # Map X-axis to 2 columns (0, 1) - boundary at 0.5
    if gaze_x < 0.5:
        col = 0
    else:
        col = 1
        
    # Map Y-axis to 2 rows (0, 1) - boundary at 0.5
    if gaze_y < 0.5:
        row = 0
    else:
        row = 1
        
    # Calculate final 0-3 index: Index = (Row * 2) + Column
    detected_index = (row * 2) + col
    return detected_index


# --- Note: The Dwell and Transition Logic (State Machine) ---
# The logic to track 'current_card', 'gaze_start_time', 'DWELL_TIME', and 'TRANSITION_TIME' 
# is essential for the *application's behavior* but is implementation-specific 
# (i.e., it depends on how you handle time and state machine transitions in Flask). 
# It is not included here, as it sits outside the core mathematical/CV logic above.