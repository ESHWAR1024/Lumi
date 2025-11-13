import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import font
import time
import threading

# ---------------- Mediapipe Face Mesh Setup ----------------
mp_face_mesh = mp.solutions.face_mesh
# Use static_image_mode=False for video stream, refine_landmarks=True for better iris detection
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Comprehensive landmark indices
NOSE_TIP = 1
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
FOREHEAD = 10
CHIN = 152

# ---------------- Tkinter Setup ----------------
root = tk.Tk()
root.title("Routine Cards Eye Tracker")
root.geometry("1200x400")
root.resizable(False, False) # Keep the window fixed size for consistent card positions

# Cards to display
cards = ["Wake Up", "Breakfast", "School", "Lunch", "Therapy", "Playtime", "Dinner", "Bedtime"]
card_labels = []

# Custom font
fnt = font.Font(family="Arial", size=16, weight="bold")

# Output label
output_var = tk.StringVar()
output_var.set("Look at a card to select it...")
output_label = tk.Label(root, textvariable=output_var, font=fnt, fg="#1e40af", height=2)
output_label.pack(pady=10, padx=10, fill='x')

# Grid layout for cards (2 rows, 4 columns)
cards_frame = tk.Frame(root, padx=10, pady=10, bg="#f3f4f6")
cards_frame.pack(pady=10)

for idx, card in enumerate(cards):
    row = idx // 4
    col = idx % 4
    lbl = tk.Label(
        cards_frame, 
        text=card, 
        bg="#bfdbfe", # Light Blue
        width=12, 
        height=3, 
        font=fnt, 
        relief="raised", 
        borderwidth=3,
        fg="#1f2937" # Dark Grey text
    )
    lbl.grid(row=row, column=col, padx=10, pady=10)
    card_labels.append(lbl)

# ---------------- Gaze Detection Logic ----------------
cap = cv2.VideoCapture(0)

# For dwell-time logic
current_card = None
gaze_start_time = None
# --- FINAL TWEAK: Increased dwell time for selection stability ---
DWELL_TIME = 1.5 

def get_iris_center(landmarks, iris_indices):
    """Get average iris position"""
    x = sum([landmarks[i].x for i in iris_indices]) / len(iris_indices)
    y = sum([landmarks[i].y for i in iris_indices]) / len(iris_indices)
    return x, y

def calculate_combined_gaze(landmarks):
    """
    Combine head pose and eye position with OPTIMIZED factors for minimal effort AND stability.
    X-Axis: Eye Focus (10.0 sensitivity).
    Y-Axis: Head Tilt Focus (15.0 sensitivity) - Reduced from 20.0 for stability.
    """
    # Get face reference points
    nose = landmarks[NOSE_TIP]
    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]
    left_eye_inner = landmarks[LEFT_EYE_INNER]
    right_eye_inner = landmarks[RIGHT_EYE_INNER]
    left_eye_outer = landmarks[LEFT_EYE_OUTER]
    right_eye_outer = landmarks[RIGHT_EYE_OUTER]
    
    # Get iris positions
    left_iris_x, left_iris_y = get_iris_center(landmarks, LEFT_IRIS)
    right_iris_x, right_iris_y = get_iris_center(landmarks, RIGHT_IRIS)
    
    # === HORIZONTAL DETECTION (Left-Right) - Eye Focus ===
    face_width = abs(left_eye_outer.x - right_eye_outer.x)
    face_center_x = (left_eye_inner.x + right_eye_inner.x) / 2
    
    left_eye_center_x = (left_eye_inner.x + left_eye_outer.x) / 2
    right_eye_center_x = (right_eye_inner.x + right_eye_outer.x) / 2
    
    left_gaze_offset = (left_iris_x - left_eye_center_x) / face_width
    right_gaze_offset = (right_iris_x - right_eye_center_x) / face_width
    eye_gaze_x = (left_gaze_offset + right_gaze_offset) / 2
    
    head_offset_x = (nose.x - face_center_x) / face_width
    
    # X-Axis Raw: Heavily favor eye gaze (0.7)
    combined_x_raw = 0.5 + (eye_gaze_x * 0.7) + (head_offset_x * 0.3)
    
    # Horizontal Sensitivity: 10.0 (Reported as perfect)
    combined_x = 0.5 + (combined_x_raw - 0.5) * 10.0
    combined_x = max(0, min(1, combined_x)) # Clamp to 0-1 range
    
    # === VERTICAL DETECTION (Up-Down) - HEAD TILT Focus ===
    face_height = abs(forehead.y - chin.y)
    face_center_y = (forehead.y + chin.y) / 2
    
    iris_y = (left_iris_y + right_iris_y) / 2
    
    eye_gaze_y = (iris_y - face_center_y) / face_height
    
    head_offset_y = (nose.y - face_center_y) / face_height
    
    # Y-Axis Raw: Heavily favor HEAD TILT (0.7) for stability
    combined_y_raw = 0.5 + (eye_gaze_y * 0.3) + (head_offset_y * 0.7)
    
    # --- FINAL TWEAK: Vertical Sensitivity reduced to 15.0 for stability ---
    combined_y = 0.5 + (combined_y_raw - 0.5) * 15.0
    combined_y = max(0, min(1, combined_y)) # Clamp to 0-1 range
    
    return combined_x, combined_y

def map_gaze_to_card(gaze_x, gaze_y):
    """
    Map gaze to card with ULTRA-LOW vertical threshold (0.12).
    """
    
    # Vertical: The ultra-low threshold for minimal tilt activation.
    if gaze_y < 0.11: 
        row = 0  # Top row (Up/Straight)
    else:
        row = 1  # Bottom row (Down)
    
    # Horizontal: Four equal zones (0.25, 0.50, 0.75)
    if gaze_x < 0.35:
        col = 0  # Far left
    elif gaze_x < 0.50:
        col = 1  # Left-center
    elif gaze_x < 0.75:
        col = 2  # Right-center
    else:
        col = 3  # Far right
    
    card_idx = row * 4 + col
    return card_idx

def reset_card_highlights():
    """Reset all cards to default color"""
    for lbl in card_labels:
        lbl.config(bg="#bfdbfe")

def detect_gaze():
    global current_card, gaze_start_time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue
        
        # Mirror the frame so it's natural for the user to view
        frame = cv2.flip(frame, 1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate combined gaze (head + eyes)
            gaze_x, gaze_y = calculate_combined_gaze(landmarks)
            
            # Map to card
            detected_card = map_gaze_to_card(gaze_x, gaze_y)
            
            # Visual feedback on webcam
            nose = landmarks[NOSE_TIP]
            left_iris_x, left_iris_y = get_iris_center(landmarks, LEFT_IRIS)
            right_iris_x, right_iris_y = get_iris_center(landmarks, RIGHT_IRIS)
            
            # Draw crosshair/markers
            cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(left_iris_x * w), int(left_iris_y * h)), 3, (255, 0, 0), -1)
            cv2.circle(frame, (int(right_iris_x * w), int(right_iris_y * h)), 3, (255, 0, 0), -1)
            
            # Dwell-time logic
            if current_card != detected_card:
                current_card = detected_card
                gaze_start_time = time.time()
                
                reset_card_highlights()
                card_labels[detected_card].config(bg="#fcd34d") # Yellow for hovering
                
            else:
                elapsed = time.time() - gaze_start_time
                
                if elapsed >= DWELL_TIME:
                    output_var.set(f"✓ SELECTED: {cards[detected_card]}")
                    card_labels[detected_card].config(bg="#a7f3d0") # Light Green for selection
                    
                    # Pause selection briefly after a successful choice
                    time.sleep(0.7) 
                    current_card = None
                    gaze_start_time = None
                    reset_card_highlights()
                else:
                    progress = int((elapsed / DWELL_TIME) * 100)
                    # Display the current gaze coordinates for debugging/tuning
                    output_var.set(f"Looking at {cards[detected_card]}... {progress}% | X:{gaze_x:.2f} Y:{gaze_y:.2f}")
        else:
            current_card = None
            gaze_start_time = None
            reset_card_highlights()
            output_var.set("No face detected. Please look at the camera.")
            
        cv2.imshow("Webcam - Minimal Tilt Required", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.03)

# ---------------- Start Gaze Thread ----------------
threading.Thread(target=detect_gaze, daemon=True).start()

# Run Tkinter loop
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()