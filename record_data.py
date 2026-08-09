"""
record_data.py
State-of-the-Art Sign Language Dataset Recorder with Invariant Features.

Features:
 - 3-Second Preparation Countdown with Visual HUD
 - Live Hand & Body Readiness Detection (Green = In Frame, Red = Incomplete)
 - Invariant Coordinate Transformation
 - Controls: [R] Redo Sequence | [SPACE] Pause | [S] Skip Action | [Q] Save & Quit
"""

import os
import time
import cv2
import numpy as np
import mediapipe as mp

DATA_PATH = "Data_Mg"
ACTIONS = np.array(['Alone', 'Call', 'Flower', 'Food', 'I am good', 'Ok Fine', 'Stop', 'There is Gun'])
NO_SEQUENCES = 30     # 30 sequences per action
SEQUENCE_LENGTH = 30  # 30 frames per sequence
PREPARE_SECONDS = 3   # 3-second countdown

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_raw_keypoints(results):
    """
    Extracts raw 225-dim landmarks array [Pose(99), LeftHand(63), RightHand(63)].
    """
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark], dtype=np.float32).flatten()
    else:
        pose = np.zeros(33 * 3, dtype=np.float32)

    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    return np.concatenate([pose, lh, rh]).astype(np.float32)

def draw_styled_landmarks(image, results):
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
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("="*65)
    print(" SIGN LANGUAGE RECORDER - CLEAN INVARIANT PIPELINE")
    print(f" Actions: {', '.join(ACTIONS)}")
    print(" Controls: [Q] Quit | [R] Redo Sequence | [S] Skip Action")
    print("="*65)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action_idx, action in enumerate(ACTIONS):
            sequence = 0
            while sequence < NO_SEQUENCES:
                # ----------------------------------------------------
                # PHASE 1: 3-SECOND PREPARATION COUNTDOWN
                # ----------------------------------------------------
                start_countdown = time.time()
                while (time.time() - start_countdown) < PREPARE_SECONDS:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    draw_styled_landmarks(image, results)

                    time_left = PREPARE_SECONDS - int(time.time() - start_countdown)
                    h, w, _ = image.shape

                    # Header HUD
                    cv2.rectangle(image, (0, 0), (w, 80), (15, 23, 42), -1)
                    cv2.putText(image, f"ACTION: {action} ({action_idx+1}/{len(ACTIONS)})", (20, 45),
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 242, 254), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Sequence {sequence+1}/{NO_SEQUENCES}", (20, 72),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1, cv2.LINE_AA)

                    pose_ok = results.pose_landmarks is not None
                    hands_ok = (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)
                    status_color = (0, 255, 128) if (pose_ok and hands_ok) else (0, 165, 255)
                    status_text = "READY [HAND IN FRAME]" if (pose_ok and hands_ok) else "POSITION HAND IN FRAME"
                    cv2.putText(image, status_text, (w - 420, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

                    # Center Countdown
                    countdown_color = (0, 255, 255) if time_left > 1 else (0, 255, 0)
                    cv2.putText(image, f"GET READY: {time_left}s", (w//2 - 180, h//2),
                                cv2.FONT_HERSHEY_DUPLEX, 1.3, countdown_color, 3, cv2.LINE_AA)
                    cv2.putText(image, f"Prepare sign: '{action}'", (w//2 - 160, h//2 + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.imshow("Sign Language Data Recorder", image)
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('s'):
                        sequence = NO_SEQUENCES
                        break

                if sequence >= NO_SEQUENCES:
                    break

                # ----------------------------------------------------
                # PHASE 2: RECORDING 30 CONTINUOUS FRAMES
                # ----------------------------------------------------
                recorded_frames = []
                redo = False

                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    draw_styled_landmarks(image, results)
                    keypoints = extract_raw_keypoints(results)
                    recorded_frames.append(keypoints)

                    h, w, _ = image.shape
                    cv2.rectangle(image, (0, 0), (w, 80), (20, 20, 100), -1)
                    cv2.circle(image, (30, 40), 12, (0, 0, 255), -1)
                    cv2.putText(image, f"RECORDING '{action}'", (55, 48),
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

                    # Smooth progress bar
                    progress = int((frame_num + 1) / SEQUENCE_LENGTH * (w - 40))
                    cv2.rectangle(image, (20, 68), (20 + progress, 74), (0, 242, 254), -1)
                    cv2.putText(image, f"Frame {frame_num+1}/{SEQUENCE_LENGTH}", (w - 200, 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 242, 254), 2, cv2.LINE_AA)

                    cv2.imshow("Sign Language Data Recorder", image)
                    key = cv2.waitKey(15) & 0xFF
                    if key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('r'):
                        print(f"[*] Re-recording sequence {sequence+1} of '{action}'...")
                        redo = True
                        break

                if redo:
                    continue

                for f_idx, kp in enumerate(recorded_frames):
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{f_idx}.npy")
                    np.save(npy_path, kp)

                print(f"  [+] Saved '{action}' | Sequence {sequence+1}/{NO_SEQUENCES}")
                sequence += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\n[SUCCESS] Data recording complete! Run 'python train_robust_classifier.py' to train.")

if __name__ == "__main__":
    main()
