"""vyton.body3d.photo_to_self — 形式①:照片→虚拟自己
链:MediaPipe 33关键点 → measure_head v2(身高/体重/性别) → 补全 → gen_virtual_self
多照片(三视图)→融合(multi_frame_fusion思想:中位数)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

_MP, _HEAD = None, None

def _kps_from_photo(img_path):
    """单照→33关键点(归一化特征,与train_v2同构)"""
    global _MP
    if _MP is None:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        _MP = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=r"B:\mode\models\pose_landmarker_lite.task")))
    from mediapipe import Image as MPImage, ImageFormat
    from PIL import Image as PILImage
    img = np.asarray(PILImage.open(img_path).convert("RGB"))
    mp_img = MPImage(image_format=ImageFormat.SRGB, data=img)
    res = _MP.detect(mp_img)
    if not res.pose_landmarks:
        return None
    lm = np.array([[p.x, p.y] for p in res.pose_landmarks[0]], np.float32)  # 33x2
    return lm

def _feats(lm):
    """与train_v2同构的归一化特征(腰原点/身高单位)"""
    # 关键索引:11左肩12右肩23左髋24右髋27左踝28右踝0鼻
    c = (lm[23] + lm[24]) / 2                       # 髋中点(与train_v2一致)
    span = np.linalg.norm(lm[10] - lm[31]) + 1e-6   # 关键点10-31距离
    out = (lm - c) / span
    return out.reshape(-1)

def _predict(feats):
    """measure_head v2 → height/weight/sex"""
    global _HEAD
    import torch, torch.nn as nn
    if _HEAD is None:
        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Sequential(torch.nn.Linear(66, 128), torch.nn.ReLU(),
                                                    torch.nn.Linear(128, 64), torch.nn.ReLU())
                self.reg = torch.nn.Linear(64, 2)   # 相对height/weight(0-1域)
                self.gen = torch.nn.Linear(64, 2)
            def forward(self, x):
                f = self.backbone(x)
                return self.reg(f), self.gen(f)
        _HEAD = Head()
        _HEAD.load_state_dict(torch.load(r"B:\mode\measure\measure_head_v2.pt", map_location="cpu"))
        _HEAD.eval()
    with torch.no_grad():
        reg, sex = _HEAD(torch.from_numpy(feats.astype(np.float32)).unsqueeze(0))
    # v2输出=相对值(0.625-0.726身高域;体重同)→换算米/kg
    h_rel, w_rel = reg[0].tolist()
    g = int(sex[0].argmax())
    return h_rel, w_rel, g

def photo_to_self(photo_paths, height=None, weight=None, sex=None, **kw):
    """形式①:照片(1-3张)±身高体重→虚拟自己(已知身高体重优先=校准)"""
    preds = []
    for p in photo_paths:
        lm = _kps_from_photo(p)
        if lm is not None:
            preds.append(_predict(_feats(lm)))
    if not preds:
        # 识别失败→纯补全
        from body3d.virtual_self import gen_virtual_self
        return gen_virtual_self(height=height, weight=weight, sex=sex, **kw)
    hs = np.median([p[0] for p in preds])
    ws = np.median([p[1] for p in preds])
    gs = int(np.median([p[2] for p in preds]).round())
    # 相对域→绝对(标定:0.625→1.50m, 0.726→2.00m;体重0.48→40kg,0.90→110kg)
    ph = 1.50 + (float(hs) - 0.625) / (0.726 - 0.625) * 0.50
    pw = 40 + (float(ws) - 0.48) / (0.90 - 0.48) * 70
    ps = "f" if gs == 1 else "m"
    # 用户给的真实值优先(校准测量)
    H = height if height else (ph if 1.3 < ph < 2.2 else 172)
    W = weight if weight else (pw if 30 < pw < 200 else 70)
    S = sex if sex else ps
    from body3d.virtual_self import gen_virtual_self
    mesh, rep = gen_virtual_self(height=H, weight=W, sex=S, **kw)
    rep["photo_measure"] = dict(pred_height=round(ph, 2), pred_weight=round(pw, 1),
                                pred_sex=ps, n_photos=len(preds),
                                calibrated=bool(height or weight))
    return mesh, rep
