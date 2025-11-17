"""
Eye Tracking Service for Picture Card Selection
Uses MediaPipe Face Mesh for gaze detection
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
import asyncio
from typing import Optional
import uvicorn
from collections import deque

app = FastAPI(title="Lumi Eye Tracking Service", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe Face Mesh Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, 
    min_detection_confidence=0.6,  # Increased for better face detection
    min_tracking_confidence=0.6     # Better tracking stability
)

# Landmark indices
NOSE_TIP = 1
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
FOREHEAD = 10
CHIN = 152

# Gaze tracking state
current_card: Optional[int] = None
gaze_start_time: Optional[float] = None
DWELL_TIME = 5.0  # 5 seconds to select (more deliberate)

# IMPROVED: Transition logic to prevent accidental selections
transition_card: Optional[int] = None
transition_start_time: Optional[float] = None
TRANSITION_TIME = 0.2  # INCREASED SPEED: Faster transitions (was 0.3)

# OPTIMIZED FOR MAXIMUM ACCURACY
CALIBRATION_OFFSET_X = 0.0  # No offset - let user's natural gaze be center
CALIBRATION_OFFSET_Y = -0.03  # Reduced upward bias for better vertical accuracy
SMOOTHING_FACTOR_ALPHA = 0.35  # Balanced smoothing (not too fast, not too slow)
SENSITIVITY_X = 22.0  # Increased horizontal sensitivity for better left/right detection
SENSITIVITY_Y = 20.0  # Increased vertical sensitivity for better card distinction

# Multi-frame smoothing for better accuracy
from collections import deque
SMOOTHING_WINDOW = 7  # Increased to 7 frames for smoother tracking
gaze_history_x = deque(maxlen=SMOOTHING_WINDOW)
gaze_history_y = deque(maxlen=SMOOTHING_WINDOW)

# Smoothing variables
smoothed_x = 0.5
smoothed_y = 0.5


def get_iris_center(landmarks, iris_indices):
    """Get average iris position"""
    x = sum([landmarks[i].x for i in iris_indices]) / len(iris_indices)
    y = sum([landmarks[i].y for i in iris_indices]) / len(iris_indices)
    return x, y


def calculate_combined_gaze(landmarks):
    """
    IMPROVED: Combine head pose and eye position for gaze detection with smoothing
    Returns (x, y) coordinates normalized to 0-1
    Based on newtrack.py improvements
    """
    global smoothed_x, smoothed_y
    
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
    
    # X-Fusion: OPTIMIZED weights for accuracy (eye-dominant for horizontal)
    combined_x_raw = 0.5 + eye_gaze_x * 0.85 + head_offset_x * 0.15
    combined_x = 0.5 + (combined_x_raw - 0.5) * SENSITIVITY_X - CALIBRATION_OFFSET_X
    combined_x = max(0.0, min(1.0, combined_x))
    
    # ---------------- 2. Y-AXIS (Vertical, Head-Dominant Logic) ----------------
    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]
    
    face_height = abs(forehead.y - chin.y)
    if face_height < 0.01:  # Prevent division by zero
        face_height = 0.01
    
    face_center_y = (forehead.y + chin.y) / 2
    iris_y = (left_iris_y + right_iris_y) / 2
    
    eye_gaze_y = (iris_y - face_center_y) / face_height
    head_offset_y = (nose.y - face_center_y) / face_height
    
    # Y-Fusion: OPTIMIZED for vertical accuracy (head-dominant for vertical)
    combined_y_raw = 0.5 + (eye_gaze_y * 0.40) + (head_offset_y * 0.60)
    combined_y = 0.5 + (combined_y_raw - 0.5) * SENSITIVITY_Y - CALIBRATION_OFFSET_Y
    combined_y = max(0.0, min(1.0, combined_y))
    
    # ---------------- 3. ENHANCED Multi-Frame Smoothing ----------------
    gaze_history_x.append(combined_x)
    gaze_history_y.append(combined_y)
    
    # Weighted moving average (recent frames weighted more)
    if len(gaze_history_x) >= 3:
        weights = list(range(1, len(gaze_history_x) + 1))  # [1, 2, 3, 4, 5]
        smoothed_x = sum(x * w for x, w in zip(gaze_history_x, weights)) / sum(weights)
        smoothed_y = sum(y * w for y, w in zip(gaze_history_y, weights)) / sum(weights)
    else:
        smoothed_x = combined_x
        smoothed_y = combined_y
    
    return smoothed_x, smoothed_y


def map_gaze_to_card(gaze_x, gaze_y, num_cards=4):
    """
    OPTIMIZED: Map gaze to card with adaptive boundaries and hysteresis
    Cards arranged vertically: [0] [1] [2] [3]
    
    Uses optimized zone boundaries that match natural gaze distribution
    with hysteresis to prevent jitter at boundaries.
    """
    global current_card
    
    # OPTIMIZED zone boundaries (not equal - matches natural gaze)
    # Top card gets slightly smaller zone, middle cards get more space
    boundaries = [0.24, 0.50, 0.76]  # Fine-tuned for better card separation
    
    # Apply hysteresis if we have a current card (prevents flickering)
    HYSTERESIS = 0.08  # Increased sticky zones for more stable selection
    
    if current_card is not None:
        if current_card == 0:  # Top card - make upper boundary sticky
            boundaries[0] += HYSTERESIS
        elif current_card == 1:  # Second card - make both boundaries sticky
            boundaries[0] -= HYSTERESIS
            boundaries[1] += HYSTERESIS
        elif current_card == 2:  # Third card - make both boundaries sticky
            boundaries[1] -= HYSTERESIS
            boundaries[2] += HYSTERESIS
        elif current_card == 3:  # Bottom card - make lower boundary sticky
            boundaries[2] -= HYSTERESIS
    
    # Map Y coordinate to card index
    if gaze_y < boundaries[0]:
        return 0  # Top card
    elif gaze_y < boundaries[1]:
        return 1  # Second card
    elif gaze_y < boundaries[2]:
        return 2  # Third card
    else:
        return 3  # Bottom card


def map_gaze_to_button(gaze_x, gaze_y):
    """
    Map gaze to action buttons (Proceed to Solution or Dig Deeper)
    Buttons arranged horizontally: [4=Proceed] [5=Dig Deeper]
    
    Returns button index (4 or 5) or None if not looking at buttons
    """
    global current_card
    
    # Buttons are in the center-bottom area
    # Y threshold: buttons are in lower portion (y > 0.55)
    # X boundaries: left (0.2-0.5), right (0.5-0.8)
    
    BUTTON_Y_THRESHOLD = 0.55  # Buttons start at 55% down the screen
    LEFT_BOUNDARY = 0.35  # Left button zone
    RIGHT_BOUNDARY = 0.65  # Right button zone
    HYSTERESIS_X = 0.05  # Reduced hysteresis for better accuracy
    
    # Only consider button zone if looking low enough
    if gaze_y < BUTTON_Y_THRESHOLD:
        return None
    
    # Apply hysteresis based on current selection
    left_bound = LEFT_BOUNDARY
    right_bound = RIGHT_BOUNDARY
    
    if current_card == 4:  # Currently on left button
        right_bound -= HYSTERESIS_X  # Make it easier to stay on left
    elif current_card == 5:  # Currently on right button
        left_bound += HYSTERESIS_X  # Make it easier to stay on right
    
    # Determine which button with clear zones
    if gaze_x < left_bound:
        return 4  # Proceed to Solution (left)
    elif gaze_x > right_bound:
        return 5  # Dig Deeper (right)
    else:
        # In the middle zone - use simple center split
        return 4 if gaze_x < 0.5 else 5


def map_gaze_to_solution_button(gaze_x, gaze_y):
    """
    Map gaze to solution feedback buttons (This Helps or Try Again)
    Buttons arranged horizontally: [6=This Helps] [7=Try Again]
    
    Returns button index (6 or 7) or None if not looking at buttons
    """
    global current_card
    
    # Solution buttons are in the center-bottom area
    # Y threshold: buttons are in lower portion (y > 0.55)
    # X boundaries: left (0.2-0.5), right (0.5-0.8)
    
    BUTTON_Y_THRESHOLD = 0.55  # Buttons start at 55% down the screen
    LEFT_BOUNDARY = 0.35  # Left button zone
    RIGHT_BOUNDARY = 0.65  # Right button zone
    HYSTERESIS_X = 0.05  # Reduced hysteresis for better accuracy
    
    # Only consider button zone if looking low enough
    if gaze_y < BUTTON_Y_THRESHOLD:
        return None
    
    # Apply hysteresis based on current selection
    left_bound = LEFT_BOUNDARY
    right_bound = RIGHT_BOUNDARY
    
    if current_card == 6:  # Currently on left button
        right_bound -= HYSTERESIS_X  # Make it easier to stay on left
    elif current_card == 7:  # Currently on right button
        left_bound += HYSTERESIS_X  # Make it easier to stay on right
    
    # Determine which button with clear zones
    if gaze_x < left_bound:
        return 6  # This Helps (left)
    elif gaze_x > right_bound:
        return 7  # Try Again (right)
    else:
        # In the middle zone - use simple center split
        return 6 if gaze_x < 0.5 else 7


def map_gaze_to_regenerate_button(gaze_x, gaze_y):
    """
    Map gaze to regenerate button (centered below cards)
    Button index: 8 = Regenerate
    
    Returns 8 if looking at regenerate button, None otherwise
    """
    # Regenerate button is centered at bottom
    # Y threshold: very bottom of screen (y > 0.75)
    # X range: centered (0.3 - 0.7)
    
    BUTTON_Y_THRESHOLD = 0.75  # Button at very bottom
    X_LEFT = 0.3
    X_RIGHT = 0.7
    
    # Check if looking at button zone
    if gaze_y > BUTTON_Y_THRESHOLD and X_LEFT < gaze_x < X_RIGHT:
        return 8  # Regenerate button
    
    return None


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Lumi Eye Tracking Service",
        "version": "1.0.0"
    }


@app.websocket("/ws/eye-tracking")
async def eye_tracking_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time eye tracking
    Receives video frames, returns gaze data
    """
    await websocket.accept()
    print("✅ Eye tracking WebSocket connected")
    
    global current_card, gaze_start_time, transition_card, transition_start_time
    current_card = None
    gaze_start_time = None
    transition_card = None
    transition_start_time = None
    
    try:
        while True:
            # Receive frame from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "frame":
                # Decode base64 image
                frame_data = message.get("frame")
                mode = message.get("mode", "cards")  # "cards" or "buttons"
                frame_bytes = base64.b64decode(frame_data.split(",")[1])
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Calculate gaze
                    gaze_x, gaze_y = calculate_combined_gaze(landmarks)
                    
                    # Map to card or button based on mode
                    if mode == "buttons":
                        detected_card = map_gaze_to_button(gaze_x, gaze_y)
                        # If not looking at buttons, keep current selection
                        if detected_card is None:
                            detected_card = current_card if current_card is not None else 4
                    elif mode == "solution":
                        detected_card = map_gaze_to_solution_button(gaze_x, gaze_y)
                        # If not looking at solution buttons, keep current selection
                        if detected_card is None:
                            detected_card = current_card if current_card is not None else 6
                    elif mode == "cards_with_regenerate":
                        # First check if looking at regenerate button
                        regenerate_detected = map_gaze_to_regenerate_button(gaze_x, gaze_y)
                        if regenerate_detected is not None:
                            detected_card = regenerate_detected
                        else:
                            # Otherwise map to cards
                            detected_card = map_gaze_to_card(gaze_x, gaze_y, num_cards=4)
                    else:
                        detected_card = map_gaze_to_card(gaze_x, gaze_y, num_cards=4)
                    
                    # IMPROVED: Dwell and Transition Logic
                    import time
                    current_time = time.time()
                    progress = 0
                    
                    if current_card is None:
                        # First detection
                        current_card = detected_card
                        gaze_start_time = current_time
                        progress = 0
                        
                    elif current_card != detected_card:
                        # Looking at different card - start transition
                        if transition_card != detected_card:
                            # New transition target
                            transition_card = detected_card
                            transition_start_time = current_time
                            progress = 0
                        else:
                            # Continuing to look at transition target
                            elapsed = current_time - transition_start_time
                            if elapsed >= TRANSITION_TIME:
                                # Transition confirmed - switch cards
                                current_card = detected_card
                                gaze_start_time = current_time
                                transition_card = None
                                transition_start_time = None
                                progress = 0
                            else:
                                # Still in transition
                                progress = 0
                    else:
                        # Looking at current card - check for selection
                        transition_card = None
                        transition_start_time = None
                        
                        if gaze_start_time:
                            elapsed = current_time - gaze_start_time
                            progress = min(100, int((elapsed / DWELL_TIME) * 100))
                            
                            if elapsed >= DWELL_TIME:
                                # Selection complete!
                                await websocket.send_json({
                                    "type": "selection",
                                    "card_index": detected_card,
                                    "gaze_x": gaze_x,
                                    "gaze_y": gaze_y
                                })
                                current_card = None
                                gaze_start_time = None
                                continue
                    
                    # Send gaze data
                    await websocket.send_json({
                        "type": "gaze",
                        "card_index": current_card if current_card is not None else detected_card,
                        "progress": progress,
                        "gaze_x": gaze_x,
                        "gaze_y": gaze_y,
                        "face_detected": True
                    })
                else:
                    # No face detected - reset everything
                    current_card = None
                    gaze_start_time = None
                    transition_card = None
                    transition_start_time = None
                    await websocket.send_json({
                        "type": "gaze",
                        "face_detected": False
                    })
            
            await asyncio.sleep(0.03)  # ~30 FPS
            
    except WebSocketDisconnect:
        print("❌ Eye tracking WebSocket disconnected")
    except Exception as e:
        print(f"❌ Error in eye tracking: {e}")
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8002)
