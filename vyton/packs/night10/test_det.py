
# 测:原始e2e图的手检测(v5被修复糊了手,原始图可能检测得出)
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
opts = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=r"B:\mode\models\body3d\hand_landmarker.task"),
    num_hands=2, min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3)
det = mp_vision.HandLandmarker.create_from_options(opts)
import glob
for p in [r"B:\mode\vton_training\e2e_hand_fixed.png", r"B:\mode\vton_training\e2e_hand_inpaint.png"]:
    try:
        img = Image.open(p).convert("RGB")
        W, H = img.size
        big = img.resize((W*2, H*2), Image.LANCZOS)
        r0 = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img)))
        r1 = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(big)))
        print(p.split(chr(92))[-1], "原尺寸:", len(r0.hand_landmarks), "| 2x:", len(r1.hand_landmarks))
    except Exception as e:
        print(p, "err", str(e)[:50])
