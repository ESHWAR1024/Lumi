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
DWELL_TIME = 1.5  # seconds to select


def get_iris_center(landmarks, iris_indices):
    """Get average iris position"""
    x = sum([landmarks[i].x for i in iris_indices]) / len(iris_indices)
    y = sum([landmarks[i].y for i in iris_indices]) / len(iris_indices)
    return x, y


def calculate_combined_gaze(landmarks):
    """
    Combine head pose and eye position for gaze detection
    Returns (x, y) coordinates normalized to 0-1
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
    
    # Combine eye and head movement
    combined_x_raw = 0.5 + (eye_gaze_x * 0.7) + (head_offset_x * 0.3)
    combined_x = 0.5 + (combined_x_raw - 0.5) * 10.0
    combined_x = max(0, min(1, combined_x))
    
    # === VERTICAL DETECTION (Up-Down) ===
    face_height = abs(forehead.y - chin.y)
    face_center_y = (forehead.y + chin.y) / 2
    
    iris_y = (left_iris_y + right_iris_y) / 2
    eye_gaze_y = (iris_y - face_center_y) / face_height
    head_offset_y = (nose.y - face_center_y) / face_height
    
    # Combine with head tilt focus
    combined_y_raw = 0.5 + (eye_gaze_y * 0.3) + (head_offset_y * 0.7)
    combined_y = 0.5 + (combined_y_raw - 0.5) * 15.0
    combined_y = max(0, min(1, combined_y))
    
    return combined_x, combined_y


def map_gaze_to_card(gaze_x, gaze_y, num_cards=4):
    """
    Map gaze coordinates to card index
    Assumes cards are in a single row (4 cards)
    """
    # For 4 cards in a row
    if gaze_x < 0.25:
        return 0
    elif gaze_x < 0.50:
        return 1
    elif gaze_x < 0.75:
        return 2
    else:
        return 3


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
    
    global current_card, gaze_start_time
    current_card = None
    gaze_start_time = None
    
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
                    
                    # Map to card (4 cards)
                    detected_card = map_gaze_to_card(gaze_x, gaze_y, num_cards=4)
                    
                    # Dwell time logic
                    import time
                    current_time = time.time()
                    
                    if current_card != detected_card:
                        current_card = detected_card
                        gaze_start_time = current_time
                        progress = 0
                    else:
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
                        else:
                            progress = 0
                    
                    # Send gaze data
                    await websocket.send_json({
                        "type": "gaze",
                        "card_index": detected_card,
                        "progress": progress,
                        "gaze_x": gaze_x,
                        "gaze_y": gaze_y,
                        "face_detected": True
                    })
                else:
                    # No face detected
                    current_card = None
                    gaze_start_time = None
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
