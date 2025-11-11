"""
Real-time emotion recognition from webcam or video files.
Usage:
    python realtime_inference.py --source webcam
    python realtime_inference.py --source video --video_path path/to/video.mp4
    python realtime_inference.py --source image --image_path path/to/image.jpg
"""

import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from src.inference import load_model, predict_from_pil
from src.dataset import CLASS_NAMES
import time


class EmotionDetector:
    def __init__(self, model_path, device='cpu', img_size=48):
        """Initialize emotion detector with trained model."""
        self.model = load_model(model_path, device=device, arch='resnet')
        self.device = device
        self.img_size = img_size
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print(f"✅ Model loaded on {device}")
    
    def detect_faces(self, frame):
        """Detect faces in frame using Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image."""
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        result = predict_from_pil(self.model, pil_img, self.device, self.img_size)
        return result
    
    def draw_results(self, frame, faces, predictions):
        """Draw bounding boxes and emotion labels on frame."""
        for (x, y, w, h), pred in zip(faces, predictions):
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Prepare text
            label = pred['label']
            confidence = pred['prob']
            text = f"{label}: {confidence:.2f}"
            
            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y-30), (x+text_w, y), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def process_webcam(self, camera_id=0, save_output=False):
        """Process webcam feed in real-time."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("✅ Webcam opened successfully")
        print("Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Predict emotions for each face
            predictions = []
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                try:
                    pred = self.predict_emotion(face_roi)
                    predictions.append(pred)
                except Exception as e:
                    print(f"Prediction error: {e}")
                    predictions.append({'label': 'Error', 'prob': 0.0})
            
            # Draw results
            frame = self.draw_results(frame, faces, predictions)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Emotion Recognition - Webcam', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam closed")
    
    def process_video(self, video_path, save_output=False, output_path=None):
        """Process video file."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"✅ Video loaded: {video_path}")
        print(f"   Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer if saving output
        writer = None
        if save_output:
            if output_path is None:
                output_path = f"output_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Saving output to: {output_path}")
        
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Predict emotions
            predictions = []
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                try:
                    pred = self.predict_emotion(face_roi)
                    predictions.append(pred)
                except:
                    predictions.append({'label': 'Error', 'prob': 0.0})
            
            # Draw results
            frame = self.draw_results(frame, faces, predictions)
            
            # Progress indicator
            cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame if saving
            if writer:
                writer.write(frame)
            
            # Display frame
            cv2.imshow('Emotion Recognition - Video', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⚠️ Processing interrupted by user")
                break
        
        cap.release()
        if writer:
            writer.release()
            print(f"✅ Output saved to: {output_path}")
        cv2.destroyAllWindows()
        print("✅ Video processing complete")
    
    def process_image(self, image_path):
        """Process single image."""
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"❌ Error: Could not load image: {image_path}")
            return
        
        print(f"✅ Image loaded: {image_path}")
        
        # Detect faces
        faces = self.detect_faces(frame)
        print(f"   Detected {len(faces)} face(s)")
        
        # Predict emotions
        predictions = []
        for idx, (x, y, w, h) in enumerate(faces):
            face_roi = frame[y:y+h, x:x+w]
            pred = self.predict_emotion(face_roi)
            predictions.append(pred)
            print(f"   Face {idx+1}: {pred['label']} ({pred['prob']:.2%})")
        
        # Draw results
        frame = self.draw_results(frame, faces, predictions)
        
        # Save result
        output_path = f"output_{int(time.time())}.jpg"
        cv2.imwrite(output_path, frame)
        print(f"💾 Result saved to: {output_path}")
        
        # Display result
        cv2.imshow('Emotion Recognition - Image', frame)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Real-time Emotion Recognition')
    parser.add_argument('--model_path', type=str, 
                       default='models/checkpoints/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--source', type=str, choices=['webcam', 'video', 'image'],
                       default='webcam', help='Input source type')
    parser.add_argument('--video_path', type=str, help='Path to video file')
    parser.add_argument('--image_path', type=str, help='Path to image file')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera device ID')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--save_output', action='store_true', 
                       help='Save processed video/image')
    parser.add_argument('--output_path', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = EmotionDetector(args.model_path, device=args.device)
    
    # Process based on source type
    if args.source == 'webcam':
        detector.process_webcam(camera_id=args.camera_id, save_output=args.save_output)
    elif args.source == 'video':
        if not args.video_path:
            print("❌ Error: --video_path required for video source")
            return
        detector.process_video(args.video_path, save_output=args.save_output, 
                              output_path=args.output_path)
    elif args.source == 'image':
        if not args.image_path:
            print("❌ Error: --image_path required for image source")
            return
        detector.process_image(args.image_path)


if __name__ == '__main__':
    main()