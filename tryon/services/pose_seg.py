# tryon/services/pose_seg.py
import json
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

class PoseSegError(RuntimeError):
    pass

def process_user_image(person_image_path, out_dir):
    """
    Extract pose (17 keypoints) and person mask using MediaPipe.
    Strict mode: if anything is missing or invalid -> raises PoseSegError.
    Returns (pose_json_path, body_mask_path) as str.
    """
    person_image_path = Path(person_image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pose_json_path = out_dir / f"{person_image_path.stem}_pose.json"
    body_mask_path = out_dir / f"{person_image_path.stem}_body_mask.png"

    bgr = cv2.imread(str(person_image_path))
    if bgr is None:
        raise PoseSegError(f"Cannot read image: {person_image_path}")
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=True) as pose:
        res = pose.process(rgb)

    # Pose landmarks strictly required
    if not res.pose_landmarks:
        raise PoseSegError("Pose landmarks not detected. Ensure the person is fully visible and well lit.")

    lm = res.pose_landmarks.landmark
    def pt(i): return [int(lm[i].x*w), int(lm[i].y*h), float(getattr(lm[i], "visibility", 1.0))]

    kp = [
        pt(mp_pose.PoseLandmark.NOSE.value),
        pt(mp_pose.PoseLandmark.LEFT_EYE.value),
        pt(mp_pose.PoseLandmark.RIGHT_EYE.value),
        pt(mp_pose.PoseLandmark.LEFT_EAR.value),
        pt(mp_pose.PoseLandmark.RIGHT_EAR.value),
        pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
        pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
        pt(mp_pose.PoseLandmark.LEFT_ELBOW.value),
        pt(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
        pt(mp_pose.PoseLandmark.LEFT_WRIST.value),
        pt(mp_pose.PoseLandmark.RIGHT_WRIST.value),
        pt(mp_pose.PoseLandmark.LEFT_HIP.value),
        pt(mp_pose.PoseLandmark.RIGHT_HIP.value),
        pt(mp_pose.PoseLandmark.LEFT_KNEE.value),
        pt(mp_pose.PoseLandmark.RIGHT_KNEE.value),
        pt(mp_pose.PoseLandmark.LEFT_ANKLE.value),
        pt(mp_pose.PoseLandmark.RIGHT_ANKLE.value),
    ]
    data = {"keypoints": kp, "image_size": [w, h], "pose_model": "MediaPipePose"}
    with open(pose_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Segmentation strictly required
    if getattr(res, "segmentation_mask", None) is None:
        raise PoseSegError("Segmentation mask not produced by MediaPipe.")

    m = (res.segmentation_mask > 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    coverage = float(np.sum(m > 0)) / float(w*h)
    if coverage < 0.1 or coverage > 0.95:
        raise PoseSegError(f"Invalid segmentation coverage ({coverage:.2f}). Retake the photo with a clearer person/background.")

    cv2.imwrite(str(body_mask_path), m)
    return str(pose_json_path), str(body_mask_path)
