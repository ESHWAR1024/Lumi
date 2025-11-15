"""
Real-time emotion recognition (webcam, video, image)

Run BEST FER+ model:
python realtime_inference.py --arch best --model_path models/ferplus_checkpoints/best_model.pt --source webcam
"""

import cv2
import torch
import argparse
import numpy as np
from PIL import Image
import time

# ------------------------- IMPORTS ------------------------------
from src.model_advanced import get_model as get_advanced_model
from src.inference import load_model, predict_from_pil
from src.dataset import CLASS_NAMES

try:
    from src.optimal_emotion_model import get_optimal_model
except:
    get_optimal_model = None
# ---------------------------------------------------------------



# =================================================================
# EMOTION DETECTOR CLASS
# =================================================================
class EmotionDetector:

    def __init__(self, model_path, device='cpu', img_size=224, arch='efficientnet'):
        self.device = device
        self.img_size = img_size
        arch_l = arch.lower()

        # ----------------------------------------------------------
        # ADVANCED MODELS (EfficientNetV2/B3/ResNetV2)
        # ----------------------------------------------------------
        if arch_l in ['efficientnetv2', 'efficientnetb3', 'resnetv2']:
            print(f"🔱 Loading advanced model: {arch}")

            self.model = get_advanced_model(
                model_name=arch_l,
                num_classes=7,
                pretrained=False
            )

            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])

            self.model.to(device)
            self.model.eval()


        # ----------------------------------------------------------
        # OPTIMAL EmotionNet
        # ----------------------------------------------------------
        elif arch_l in ['emotionnet', 'optimal']:
            print("🔥 Loading EmotionNet (optimal)")

            if get_optimal_model is None:
                raise ValueError("optimal_emotion_model.py missing!")

            self.model = get_optimal_model(
                num_classes=7,
                dataset_size='medium',
                pretrained=False
            )

            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])

            self.model.to(device)
            self.model.eval()


        # ----------------------------------------------------------
        # BEST FER+ MODEL (EfficientNetV2, 8 CLASSES)
        # ----------------------------------------------------------
        elif arch_l == "best":
            print("🌟 Loading BEST FER+ EfficientNetV2 model")

            # Your checkpoint has 8 classes (FER+ uses 8)
            self.model = get_advanced_model(
                model_name="efficientnetv2",
                num_classes=8,        # IMPORTANT FIX
                pretrained=False
            )

            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])

            self.model.to(device)
            self.model.eval()


        # ----------------------------------------------------------
        # LEGACY MODELS
        # ----------------------------------------------------------
        else:
            print(f"📦 Loading legacy model: {arch}")
            self.model = load_model(model_path, device=device, arch=arch)


        # FACE DETECTOR
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        print(f"✅ Model loaded on {device} (arch={arch})")



    # =================================================================
    # FACE DETECTION
    # =================================================================
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces


    # =================================================================
    # EMOTION PREDICTION
    # =================================================================
    def predict_emotion(self, face_img):
        pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        return predict_from_pil(self.model, pil, self.device, self.img_size)


    # =================================================================
    # DRAW RESULTS
    # =================================================================
    def draw_results(self, frame, faces, preds):
        for (x, y, w, h), pred in zip(faces, preds):
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            txt = f"{pred['label']}: {pred['prob']:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            cv2.rectangle(frame, (x, y-30), (x+tw, y), (0,255,0), -1)
            cv2.putText(frame, txt, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        return frame



    # =================================================================
    # WEBCAM MODE
    # =================================================================
    def process_webcam(self, camera_id=0, save_output=False):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return

        print("Press 'q' to quit, 's' to screenshot.")

        fps_start = time.time()
        frame_count = 0
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.detect_faces(frame)
            preds = []

            for (x,y,w,h) in faces:
                roi = frame[y:y+h, x:x+w]
                try:
                    preds.append(self.predict_emotion(roi))
                except:
                    preds.append({'label':'Error','prob':0.0})

            frame = self.draw_results(frame, faces, preds)

            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_start)
                fps_start = time.time()

            cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

            cv2.imshow("Emotion Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                name = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(name, frame)
                print("Saved:", name)

        cap.release()
        cv2.destroyAllWindows()



    # =================================================================
    # VIDEO MODE
    # =================================================================
    def process_video(self, path, save_output=False, output_path=None):

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("❌ Cannot open video")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if save_output:
            if output_path is None:
                output_path = f"output_{int(time.time())}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w,h))
            print("Saving to:", output_path)
        else:
            writer = None

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            faces = self.detect_faces(frame)
            preds = []

            for (x,y,ww,hh) in faces:
                roi = frame[y:y+hh, x:x+ww]
                try:
                    preds.append(self.predict_emotion(roi))
                except:
                    preds.append({'label':'Error','prob':0.0})

            frame = self.draw_results(frame, faces, preds)

            cv2.putText(frame, f"Frame {frame_id}/{total}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

            if writer:
                writer.write(frame)

            cv2.imshow("Emotion Recognition - Video", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer:
            writer.release()
            print("Saved:", output_path)

        cv2.destroyAllWindows()



    # =================================================================
    # IMAGE MODE
    # =================================================================
    def process_image(self, path):

        frame = cv2.imread(path)
        if frame is None:
            print("❌ Cannot load image")
            return

        faces = self.detect_faces(frame)
        print("Faces detected:", len(faces))

        preds = []
        for i, (x,y,w,h) in enumerate(faces):
            roi = frame[y:y+h, x:x+w]
            pred = self.predict_emotion(roi)
            preds.append(pred)
            print(f"Face {i+1}: {pred['label']} ({pred['prob']:.3f})")

        frame = self.draw_results(frame, faces, preds)

        name = f"output_{int(time.time())}.jpg"
        cv2.imwrite(name, frame)
        print("Saved:", name)

        cv2.imshow("Emotion Recognition - Image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




# =================================================================
# MAIN
# =================================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str,
                        default="models/checkpoints/checkpoint_epoch50.pt")

    parser.add_argument("--source", type=str,
                        choices=["webcam","video","image"],
                        default="webcam")

    parser.add_argument("--video_path", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--camera_id", type=int, default=0)

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--save_output", action="store_true")
    parser.add_argument("--output_path", type=str)

    parser.add_argument("--arch", type=str,
                        choices=[
                            "resnet","efficientnet","simple",
                            "efficientnetv2","efficientnetb3","resnetv2",
                            "emotionnet","optimal","best"
                        ],
                        default="efficientnetv2")

    parser.add_argument("--img_size", type=int, default=224)

    args = parser.parse_args()

    detector = EmotionDetector(
        args.model_path,
        device=args.device,
        img_size=args.img_size,
        arch=args.arch
    )

    if args.source == "webcam":
        detector.process_webcam(
            camera_id=args.camera_id,
            save_output=args.save_output
        )

    elif args.source == "video":
        detector.process_video(
            args.video_path,
            save_output=args.save_output,
            output_path=args.output_path
        )

    elif args.source == "image":
        detector.process_image(args.image_path)



if __name__ == "__main__":
    main()
