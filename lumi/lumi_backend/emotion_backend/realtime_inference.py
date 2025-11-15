"""
Real-time emotion recognition from webcam or video files.

Usage:
    python realtime_inference.py --source webcam --arch efficientnet
    python realtime_inference.py --source video --video_path path/to/video.mp4 --arch efficientnet
    python realtime_inference.py --source image --image_path path/to/image.jpg --arch efficientnet
"""

import cv2
import torch
import argparse
import numpy as np
from PIL import Image
import time

from src.model_advanced import get_model as get_advanced_model
from src.inference import load_model, predict_from_pil
from src.dataset import CLASS_NAMES


class EmotionDetector:
    """
    Real-time emotion recognition class.
    """

    def __init__(self, model_path, device='cpu', img_size=224, arch='efficientnet'):
        """Initialize emotion detector with trained model."""
        self.device = device
        self.img_size = img_size

        # ---------------------------------------------------------------------
        # Load model based on architecture (Advanced models vs Legacy models)
        # ---------------------------------------------------------------------
        if arch.lower() in ['efficientnetv2', 'efficientnetb3', 'resnetv2']:
            print(f"🔱 Loading advanced architecture: {arch}")

            # Build advanced model
            self.model = get_advanced_model(
                arch_name=arch,
                num_classes=7,
                pretrained=False
            )

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state'])

            self.model.to(device)
            self.model.eval()

        else:
            print(f"📦 Loading legacy model: {arch}")
            self.model = load_model(model_path, device=device, arch=arch)

        # Haar-cascade detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        print(f"✅ Model loaded on {device} (Architecture: {arch})")

    # ------------------------------------------------------------
    # FACE DETECTION
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # EMOTION PREDICTION
    # ------------------------------------------------------------
    def predict_emotion(self, face_img):
        """Predict emotion from face image."""
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        result = predict_from_pil(self.model, pil_img, self.device, self.img_size)
        return result

    # ------------------------------------------------------------
    # DRAW RESULTS ON FRAME
    # ------------------------------------------------------------
    def draw_results(self, frame, faces, predictions):
        """Draw bounding boxes and emotion labels on frame."""
        for (x, y, w, h), pred in zip(faces, predictions):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            label = pred['label']
            confidence = pred['prob']
            text = f"{label}: {confidence:.2f}"

            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y-30), (x+text_w, y), (0, 255, 0), -1)
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    # ------------------------------------------------------------
    # REALTIME WEBCAM MODE
    # ------------------------------------------------------------
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

            faces = self.detect_faces(frame)
            predictions = []

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                try:
                    pred = self.predict_emotion(face_roi)
                except Exception as e:
                    print(f"Prediction error: {e}")
                    pred = {'label': 'Error', 'prob': 0.0}

                predictions.append(pred)

            frame = self.draw_results(frame, faces, predictions)

            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_start_time)
                fps_start_time = time.time()

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Emotion Recognition - Webcam', frame)

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

    # ------------------------------------------------------------
    # VIDEO FILE MODE
    # ------------------------------------------------------------
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
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

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

            faces = self.detect_faces(frame)
            predictions = []

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                try:
                    pred = self.predict_emotion(face_roi)
                except:
                    pred = {'label': 'Error', 'prob': 0.0}

                predictions.append(pred)

            frame = self.draw_results(frame, faces, predictions)

            cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if writer:
                writer.write(frame)

            cv2.imshow('Emotion Recognition - Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⚠️ User stopped early")
                break

        cap.release()
        if writer:
            writer.release()
            print(f"✅ Output saved to: {output_path}")

        cv2.destroyAllWindows()
        print("✅ Video processing complete")

    # ------------------------------------------------------------
    # SINGLE IMAGE MODE
    # ------------------------------------------------------------
    def process_image(self, image_path):
        """Process single image."""
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"❌ Error: Could not load image: {image_path}")
            return

        print(f"✅ Image loaded: {image_path}")

        faces = self.detect_faces(frame)
        print(f"Detected {len(faces)} face(s)")

        predictions = []
        for idx, (x, y, w, h) in enumerate(faces):
            face_roi = frame[y:y+h, x:x+w]
            pred = self.predict_emotion(face_roi)
            predictions.append(pred)

            print(f"Face {idx+1}: {pred['label']} ({pred['prob']:.2%})")
            print("All emotions:")
            for emotion, prob in sorted(pred['all_probs'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion:10s}: {prob*100:5.1f}%")

        frame = self.draw_results(frame, faces, predictions)

        output_path = f"output_{int(time.time())}.jpg"
        cv2.imwrite(output_path, frame)
        print(f"💾 Result saved to: {output_path}")

        cv2.imshow('Emotion Recognition - Image', frame)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Real-time Emotion Recognition')

    parser.add_argument('--model_path', type=str,
                       default='models/checkpoints/checkpoint_epoch50.pt',
                       help='Path to trained model checkpoint')

    parser.add_argument('--source', type=str,
                       choices=['webcam', 'video', 'image'],
                       default='webcam')

    parser.add_argument('--video_path', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--camera_id', type=int, default=0)

    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--output_path', type=str)

    parser.add_argument('--arch', type=str,
                       choices=['resnet', 'efficientnet', 'simple',
                                'efficientnetv2', 'efficientnetb3', 'resnetv2'],
                       default='efficientnet')

    parser.add_argument('--img_size', type=int, default=224)

    args = parser.parse_args()

    detector = EmotionDetector(args.model_path, device=args.device,
                               img_size=args.img_size, arch=args.arch)

    # SELECT MODE
    if args.source == 'webcam':
        detector.process_webcam(camera_id=args.camera_id,
                                save_output=args.save_output)

    elif args.source == 'video':
        if not args.video_path:
            print("❌ Error: --video_path required for video mode")
            return
        detector.process_video(args.video_path,
                               save_output=args.save_output,
                               output_path=args.output_path)

    elif args.source == 'image':
        if not args.image_path:
            print("❌ Error: --image_path required for image mode")
            return
        detector.process_image(args.image_path)


if __name__ == '__main__':
    main()
