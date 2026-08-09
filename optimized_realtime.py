"""
optimized_realtime.py
Desktop Real-Time OpenCV Sign Recognition with Invariant 162-D Skeletons.
Features:
 - MediaPipe Holistic Tracking
 - Invariant Primary Hand & Elevation Feature Extraction
 - On-Screen Live Prediction HUD & Probability Meters
"""

import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
from features import extract_invariant_features_from_raw, FEATURE_DIM
from train_robust_classifier import SignNet, ACTIONS, SEQUENCE_LENGTH

def draw_styled_landmarks(image, results, mp_holistic, mp_drawing):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(80, 44, 250), thickness=2, circle_radius=2)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 122), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=2)
        )

def main():
    MODEL_PATH = "best_sign_model.pt"
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file '{MODEL_PATH}' not found. Please run 'python train_robust_classifier.py' first.")
        return

    device = torch.device("cpu")
    model = SignNet(input_dim=FEATURE_DIM, hidden_dim=64, num_classes=len(ACTIONS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"[INFO] Loaded SignNet Invariant Model ({FEATURE_DIM} Features).")

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    prob_history = deque(maxlen=4)
    sentence = []
    threshold = 0.75

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            draw_styled_landmarks(image, results, mp_holistic, mp_drawing)

            # Extract raw landmarks
            pose_raw = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark], dtype=np.float32) if results.pose_landmarks else np.zeros((33, 3), dtype=np.float32)
            lh_raw = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark], dtype=np.float32) if results.left_hand_landmarks else np.zeros((21, 3), dtype=np.float32)
            rh_raw = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark], dtype=np.float32) if results.right_hand_landmarks else np.zeros((21, 3), dtype=np.float32)

            invariant_features = extract_invariant_features_from_raw(pose_raw, lh_raw, rh_raw)
            sequence.append(invariant_features)

            h, w, _ = image.shape
            if len(sequence) == SEQUENCE_LENGTH:
                input_tensor = torch.tensor([list(sequence)], dtype=torch.float32)
                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1).numpy()[0]
                    prob_history.append(probabilities)

                avg_probs = np.mean(prob_history, axis=0)
                pred_idx = int(np.argmax(avg_probs))
                confidence = float(avg_probs[pred_idx])

                if confidence > threshold:
                    action_label = ACTIONS[pred_idx]
                    if len(sentence) == 0 or action_label != sentence[-1]:
                        sentence.append(action_label)
                        if len(sentence) > 6:
                            sentence = sentence[-6:]

                # Probability Radar
                for idx, (action, prob) in enumerate(zip(ACTIONS, avg_probs)):
                    bar_w = int(prob * 130)
                    is_active = (idx == pred_idx and prob > threshold)
                    bar_color = (0, 242, 254) if is_active else (70, 70, 70)
                    
                    y_pos = 90 + idx * 26
                    cv2.rectangle(image, (15, y_pos), (15 + bar_w, y_pos + 18), bar_color, -1)
                    cv2.rectangle(image, (15, y_pos), (145, y_pos + 18), (100, 100, 100), 1)
                    cv2.putText(image, f"{action}: {prob*100:.0f}%", (155, y_pos + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Sentence banner
            cv2.rectangle(image, (0, 0), (w, 55), (15, 23, 42), -1)
            cv2.putText(image, ' '.join(sentence) if sentence else "Perform sign gestures...",
                        (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 242, 254), 2, cv2.LINE_AA)

            cv2.imshow("Sign Language AI (Invariant 162-D Engine)", image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
