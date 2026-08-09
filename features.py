"""
features.py
Universal Invariant Feature Extractor for Sign Language Recognition.

Extracts an invariant 162-dimensional vector per frame:
  1. Primary Hand Shape (21 joints x 3 = 63): Centered at wrist, normalized by palm length.
  2. Primary Hand Position relative to Nose (3): (Wrist - Nose) / ShoulderWidth (captures chin/ear/chest elevation).
  3. Secondary Hand Shape (21 joints x 3 = 63): Centered at wrist (or zeros if single-handed).
  4. Secondary Hand Position relative to Nose (3): (Wrist - Nose) / ShoulderWidth.
  5. Upper-Body Posture Anchors (10 joints x 3 = 30): Shoulders, elbows, chest normalized by shoulder width.
Total: 63 + 3 + 63 + 3 + 30 = 162 invariant features.
"""

import numpy as np

FEATURE_DIM = 162

def extract_hand_features(hand_landmarks):
    """
    Normalizes a hand (21 landmarks x 3) relative to its wrist (joint 0)
    and scales by palm length (joint 0 to joint 9).
    Returns (21, 3) normalized array and raw wrist position (3,).
    """
    if hand_landmarks is None or np.count_nonzero(hand_landmarks) == 0:
        return np.zeros((21, 3), dtype=np.float32), np.zeros(3, dtype=np.float32)

    hand = np.array(hand_landmarks, dtype=np.float32).reshape(21, 3)
    wrist = hand[0].copy()
    middle_knuckle = hand[9]

    palm_length = np.linalg.norm(wrist - middle_knuckle)
    scale = palm_length if palm_length > 1e-3 else 1.0

    norm_hand = (hand - wrist) / scale
    return norm_hand, wrist

def extract_invariant_features_from_raw(pose_raw, lh_raw, rh_raw):
    """
    Converts raw MediaPipe landmarks (or 225-dim vector) into the 162-dim invariant representation.
    """
    pose = np.array(pose_raw, dtype=np.float32).reshape(33, 3) if pose_raw is not None else np.zeros((33, 3), dtype=np.float32)
    lh = np.array(lh_raw, dtype=np.float32).reshape(21, 3) if lh_raw is not None else np.zeros((21, 3), dtype=np.float32)
    rh = np.array(rh_raw, dtype=np.float32).reshape(21, 3) if rh_raw is not None else np.zeros((21, 3), dtype=np.float32)

    # Upper Body Reference Points
    nose = pose[0]
    left_shoulder = pose[11]
    right_shoulder = pose[12]
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    scale_body = shoulder_width if shoulder_width > 1e-3 else 1.0

    # 1. Normalize Hands
    lh_norm, lh_wrist = extract_hand_features(lh)
    rh_norm, rh_wrist = extract_hand_features(rh)

    lh_active = np.count_nonzero(lh) > 0
    rh_active = np.count_nonzero(rh) > 0

    # Identify Primary and Secondary Hand (Ambidextrous Assignment)
    if rh_active and not lh_active:
        primary_hand = rh_norm
        primary_wrist = rh_wrist
        secondary_hand = lh_norm
        secondary_wrist = lh_wrist
    elif lh_active and not rh_active:
        primary_hand = lh_norm
        primary_wrist = lh_wrist
        secondary_hand = rh_norm
        secondary_wrist = rh_wrist
    elif rh_active and lh_active:
        primary_hand = rh_norm
        primary_wrist = rh_wrist
        secondary_hand = lh_norm
        secondary_wrist = lh_wrist
    else:
        primary_hand = np.zeros((21, 3), dtype=np.float32)
        primary_wrist = np.zeros(3, dtype=np.float32)
        secondary_hand = np.zeros((21, 3), dtype=np.float32)
        secondary_wrist = np.zeros(3, dtype=np.float32)

    # 2. Hand Position relative to Nose (Elevation & Placement Anchor)
    primary_pos = (primary_wrist - nose) / scale_body if (lh_active or rh_active) else np.zeros(3, dtype=np.float32)
    secondary_pos = (secondary_wrist - nose) / scale_body if (lh_active and rh_active) else np.zeros(3, dtype=np.float32)

    # 3. Upper Body Keypoints (10 keypoints x 3: Nose, Shoulders, Elbows, Wrists, Chest)
    body_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 8]
    body_pts = pose[body_indices]
    if np.count_nonzero(pose) > 0:
        body_pts = (body_pts - shoulder_mid) / scale_body
    else:
        body_pts = np.zeros((10, 3), dtype=np.float32)

    # Assemble 162 Invariant Feature Vector
    features = np.concatenate([
        primary_hand.flatten(),      # 63
        primary_pos.flatten(),       # 3
        secondary_hand.flatten(),    # 63
        secondary_pos.flatten(),     # 3
        body_pts.flatten()           # 30
    ]).astype(np.float32)

    return features

def extract_from_225_vector(vec225):
    """
    Extracts 162-dim invariant feature vector from a standard 225-dim [Pose(99), LH(63), RH(63)] array.
    """
    v = np.array(vec225, dtype=np.float32)
    if len(v) < 225:
        v = np.pad(v, (0, 225 - len(v)))
    
    pose = v[:99]
    lh = v[99:162]
    rh = v[162:225]
    return extract_invariant_features_from_raw(pose, lh, rh)
