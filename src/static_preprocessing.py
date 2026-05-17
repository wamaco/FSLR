from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp

HANDS = mp.solutions.hands
DRAW = mp.solutions.drawing_utils

FEATURE_SIZE = 63  # 21 landmarks * (x, y, z)


def extract_hand_landmarks_from_image(image: np.ndarray) -> np.ndarray | None:
    """Extract a normalized 63-d hand landmark feature vector.

    Returns None when no hand is detected.
    Normalization:
    - translate all points relative to wrist landmark (index 0)
    - scale by max distance from wrist to any landmark (hand size proxy)
    """
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with HANDS.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)  # [21,3]

    wrist = pts[0].copy()
    pts = pts - wrist

    dists = np.linalg.norm(pts, axis=1)
    scale = float(np.max(dists))
    if scale > 1e-8:
        pts = pts / scale

    flat = pts.reshape(-1)
    if flat.shape[0] != FEATURE_SIZE:
        return None
    return flat.astype(np.float32)


def draw_hand_landmarks(image: np.ndarray) -> np.ndarray:
    """Draw detected hand landmarks on image for preview/debug."""
    if image is None:
        return image

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with HANDS.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        results = hands.process(rgb)

    output = image.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            DRAW.draw_landmarks(output, hand_landmarks, HANDS.HAND_CONNECTIONS)
    return output
