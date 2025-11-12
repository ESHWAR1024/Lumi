import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import font
import time
import threading

# ---------------- Mediapipe Face Mesh Setup ----------------
mp_face_mesh = mp.solutions.face_mesh
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

# Cards to display
cards = ["Wake Up", "Breakfast", "School", "Lunch", "Therapy", "Playtime", "Dinner", "Bedtime"]
card_labels = []

# Custom font
fnt = font.Font(size=16, weight="bold")

# Output label
output_var = tk.StringVar()
output_var.set("Look at a card to select it...")
output_label = tk.Label(root, textvariable=output_var, font=fnt, fg="blue")
output_label.pack(pady=20)

# Grid layout for cards (2 rows, 4 columns)
cards_frame = tk.Frame(root)
cards_frame.pack(pady=20)

for idx, card in enumerate(cards):
    row = idx // 4
    col = idx % 4
    lbl = tk.Label(cards_frame, text=card, bg="lightblue", width=12, height=3, font=fnt, relief="ridge", borderwidth=3)
    lbl.grid(row=row, column=col, padx=8, pady=8)
    card_labels.append(lbl)

# ---------------- Gaze Detection Logic ----------------
cap = cv2.VideoCapture(0)

# For dwell-time logic
current_card = None
gaze_start_time = None
DWELL_TIME = 1.5  # seconds

def get_iris_center(landmarks, iris_indices):
    """Get average iris position"""
    x = sum([landmarks[i].x for i in iris_indices]) / len(iris_indices)
    y = sum([landmarks[i].y for i in iris_indices]) / len(iris_indices)
    return x, y

def calculate_combined_gaze(landmarks):
    """
    Combine head pose AND eye position with EXTREME SENSITIVITY scaling.
    Scaling factors remain high (7.0, 8.0) for minimal effort.
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
    
    # === HORIZONTAL DETECTION (Left-Right) ===
    face_width = abs(left_eye_outer.x - right_eye_outer.x)
    face_center_x = (left_eye_inner.x + right_eye_inner.x) / 2
    
    left_eye_center_x = (left_eye_inner.x + left_eye_outer.x) / 2
    right_eye_center_x = (right_eye_inner.x + right_eye_outer.x) / 2
    
    left_gaze_offset = (left_iris_x - left_eye_center_x) / face_width
    right_gaze_offset = (right_iris_x - right_eye_center_x) / face_width
    eye_gaze_x = (left_gaze_offset + right_gaze_offset) / 2
    
    head_offset_x = (nose.x - face_center_x) / face_width
    
    combined_x = 0.5 + (eye_gaze_x * 0.7) + (head_offset_x * 0.3)
    
    # *** EXTREME HORIZONTAL SENSITIVITY: 7.0 ***
    combined_x = 0.5 + (combined_x - 0.5) * 7.0
    combined_x = max(0, min(1, combined_x))  # Clamp
    
    # === VERTICAL DETECTION (Up-Down) ===
    face_height = abs(forehead.y - chin.y)
    face_center_y = (forehead.y + chin.y) / 2
    
    iris_y = (left_iris_y + right_iris_y) / 2
    
    eye_gaze_y = (iris_y - face_center_y) / face_height
    
    head_offset_y = (nose.y - face_center_y) / face_height
    
    combined_y = 0.5 + (eye_gaze_y * 0.7) + (head_offset_y * 0.3)
    
    # *** EXTREME VERTICAL SENSITIVITY: 8.0 ***
    combined_y = 0.5 + (combined_y - 0.5) * 8.0
    combined_y = max(0, min(1, combined_y))  # Clamp
    
    return combined_x, combined_y

def map_gaze_to_card(gaze_x, gaze_y):
    """
    Map gaze to card with ADJUSTED thresholds for Left Side and Vertical Access.
    """
    # Vertical: Drastically lower threshold for bottom row access (Was 0.40)
    # The new value (0.30) assumes that any slight downward movement should trigger the bottom row.
    if gaze_y < 0.30: 
        row = 0  # Top row
    else:
        row = 1  # Bottom row
    
    # Horizontal: Shift thresholds to the left to make card 0, 1 easier to access
    # Center is now around 0.55/0.65 instead of 0.45/0.55
    if gaze_x < 0.20:    # Was 0.25 (Even easier access to Far Left)
        col = 0  # Far left
    elif gaze_x < 0.40:  # Was 0.45 (Easier access to Left-center)
        col = 1  # Left-center
    elif gaze_x < 0.65:  # Wider zone for Right-center
        col = 2  # Right-center
    else:
        col = 3  # Far right (This remains the easiest to reach)
    
    card_idx = row * 4 + col
    return card_idx

def reset_card_highlights():
    """Reset all cards to default color"""
    for lbl in card_labels:
        lbl.config(bg="lightblue")

def detect_gaze():
    global current_card, gaze_start_time
    
    while True:
        ret, frame = cap.read()
        if not ret:
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
            
            # Visual feedback on webcam (Drawing code unchanged)
            nose = landmarks[NOSE_TIP]
            left_iris_x, left_iris_y = get_iris_center(landmarks, LEFT_IRIS)
            right_iris_x, right_iris_y = get_iris_center(landmarks, RIGHT_IRIS)
            
            cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(left_iris_x * w), int(left_iris_y * h)), 3, (255, 0, 0), -1)
            cv2.circle(frame, (int(right_iris_x * w), int(right_iris_y * h)), 3, (255, 0, 0), -1)
            
            # Dwell-time logic
            if current_card != detected_card:
                current_card = detected_card
                gaze_start_time = time.time()
                
                reset_card_highlights()
                card_labels[detected_card].config(bg="yellow")
                
            else:
                elapsed = time.time() - gaze_start_time
                
                if elapsed >= DWELL_TIME:
                    output_var.set(f"✓ Selected: {cards[detected_card]}")
                    card_labels[detected_card].config(bg="lightgreen")
                    
                    time.sleep(0.5)
                    current_card = None
                    gaze_start_time = None
                    reset_card_highlights()
                else:
                    progress = int((elapsed / DWELL_TIME) * 100)
                    output_var.set(f"Looking at {cards[detected_card]}... {progress}% | X:{gaze_x:.2f} Y:{gaze_y:.2f}")
        else:
            current_card = None
            gaze_start_time = None
            reset_card_highlights()
            output_var.set("No face detected. Please look at the camera.")
        
        cv2.imshow("Webcam - Just look at the card naturally!", frame)
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