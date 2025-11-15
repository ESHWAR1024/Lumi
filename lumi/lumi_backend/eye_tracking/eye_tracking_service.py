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
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5)

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
DWELL_TIME = 2.0  # seconds to select (increased for more deliberate selection)

# IMPROVED: Transition logic to prevent accidental selections
transition_card: Optional[int] = None
transition_start_time: Optional[float] = None
TRANSITION_TIME = 0.3  # seconds to confirm card change

# Improved calibration and smoothing parameters
CALIBRATION_OFFSET_X = 0.05
SMOOTHING_FACTOR_ALPHA = 0.3
SENSITIVITY = 18.0

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
    
    # X-Fusion: W_E=0.75, W_H=0.25 (IMPROVED: More eye-dominant)
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
    
    # Y-Fusion: W_E=0.3, W_H=0.7 (Head-dominant for vertical)
    combined_y_raw = 0.5 + (eye_gaze_y * 0.3) + (head_offset_y * 0.7)
    
    combined_y = 0.5 + (combined_y_raw - 0.5) * SENSITIVITY
    combined_y = max(0, min(1, combined_y))
    
    # ---------------- 3. Smoothing (IMPROVED: Alpha filter) ----------------
    smoothed_x = SMOOTHING_FACTOR_ALPHA * combined_x + (1 - SMOOTHING_FACTOR_ALPHA) * smoothed_x
    smoothed_y = SMOOTHING_FACTOR_ALPHA * combined_y + (1 - SMOOTHING_FACTOR_ALPHA) * smoothed_y
    
    return smoothed_x, smoothed_y


def map_gaze_to_card(gaze_x, gaze_y, num_cards=4):
    """
    IMPROVED: Map gaze coordinates to card index for 2x2 grid
    Cards are arranged as:
    [0] [1]
    [2] [3]
    """
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
                    
                    # Map to card (4 cards in 2x2 grid)
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
