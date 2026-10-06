
# 虚拟自己v7:性别v4先验(女:颧骨柔化+眉细+唇饱和;男:下颌方+眉浓)
import io
import torch, torch.nn as nn
from PIL import Image as PImage
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

face_img = Image.open(r"B:\mode\vton_training\face_test.png").convert("RGB")
face_arr = np.asarray(face_img).astype(np.float32)
fw, fh2 = face_img.size
lm_norm = np.load(r"B:\mode\vton_training\face_lm.npy")
flm = np.array([[p[0]*fw, p[1]*fh2] for p in lm_norm])
fx0, fx1 = flm[:,0].min(), flm[:,0].max()
fy0, fy1 = flm[:,1].min(), flm[:,1].max()

W, H2 = 560, 600
# 底层:暗化+模糊(只留高光信息,弱化"平滑照片感")
base = face_img.resize((W, H2-80))
base = base.filter(ImageFilter.GaussianBlur(6))
base = Image.eval(base, lambda p: int(p*0.25))  # 压暗到25%
img = Image.new("RGB", (W,H2), (18,18,22))
img.paste(base, (40, 50))
d = ImageDraw.Draw(img)
rng = np.random.default_rng(42)
for j in range(440):
    for i in range(440):
        uu, vv = i/439, j/439
        # 边缘硬截止(去伪影:外圈10%直接不画)
        edge = max(abs(uu-0.5), abs(vv-0.5))*2
        if edge > 0.92: continue
        sx = int(np.clip(uu*(fx1-fx0)+fx0, 0, fw-1))
        sy = int(np.clip(vv*(fy1-fy0)+fy0, 0, fh2-1))
        r_, g_, b_ = face_arr[sy, sx, :3]
        lum = (r_+g_+b_)/765
        edge_f = max(0.05, 1.0 - max(0, edge-0.5)*1.8)
        keep_p = min(1.0, 0.10 + (1.0-lum)*0.55)  # v8:噪点密度减半
        if rng.random() > keep_p * edge_f: continue
        px_ = 40 + uu*(W-80) + rng.uniform(-1.1, 1.1)
        py_ = 50 + vv*(H2-90) + rng.uniform(-1.1, 1.1)
        rad = 1.1 + (1.0-lum)*1.9
        d.ellipse((px_-rad, py_-rad, px_+rad, py_+rad),
                  fill=(int(r_*edge_f), int(g_*edge_f), int(b_*edge_f)))
# == 性别先验修正 ==
class _GN(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(3,48,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
                              nn.Conv2d(48,96,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
                              nn.Conv2d(96,192,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(192,96); s.head = nn.Linear(96,2)
    def forward(s,x): return s.head(torch.relu(s.fc(s.net(x).flatten(1))))
try:
    gm = _GN(); gm.load_state_dict(torch.load(r"B:\mode\models\vyton_gender_face_v4.pt", map_location="cpu", weights_only=True)); gm.eval()
    with torch.no_grad():
        xi = torch.tensor(np.asarray(face_img.resize((112,112)))/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0)
        pg = torch.softmax(gm(xi),1)[0]
    female = float(pg[1])
    print("gender female prob:", round(female,3), flush=True)
except Exception as e:
    female = 0.5; print("gender skip:", e, flush=True)
arr7 = np.asarray(img).astype(np.float32)
if female > 0.6:   # 女性:全局柔化+唇区饱和
    arr7 = arr7*0.97 + np.array([255,214,222],np.float32)*0.03
    arr7[...,1] *= 0.985
elif female < 0.4:  # 男性:对比+冷调
    arr7 = arr7*0.99 + np.array([230,238,250],np.float32)*0.01
    arr7[...,2] *= 1.012
# v8:发际线暗示(头顶轮廓带深棕点阵)+唇降饱和+底部压暗
if female > 0.6:
    # 唇区(下半中部)降饱和8%
    H8 = arr7.shape[0]
    lip_y0, lip_y1 = int(H8*0.62), int(H8*0.74)
    lip = arr7[lip_y0:lip_y1]
    lm_ = lip.mean(axis=2, keepdims=True)
    lip2 = lm_ + (lip - lm_)*0.92
    arr7[lip_y0:lip_y1] = lip2
    # 底部衣物压暗调暖
    arr7[int(H8*0.85):] = arr7[int(H8*0.85):]*0.82 + np.array([18,10,2],np.float32)
img = PImage.fromarray(arr7.clip(0,255).astype(np.uint8))
# 头顶发际线:沿上半轮廓画深棕点带(有序溶解)
dd = ImageDraw.Draw(img)
rng8 = np.random.default_rng(7)
H8b = img.size[1]
for t in range(140):
    uu = rng8.uniform(0.08, 0.92)
    vv = rng8.uniform(0.04, 0.30)  # 头顶带
    dens = 1.0 - abs(vv-0.16)/0.14  # 中间密两侧疏
    if rng8.random() > dens: continue
    px_ = 30 + uu*(img.size[0]-60) + rng8.uniform(-2,2)
    py_ = 30 + vv*(H8b-100) + rng8.uniform(-2,2)
    rx = 0.9 + rng8.uniform(0,0.8)   # v9:横向半径收窄
    ry = 2.2 + rng8.uniform(0,2.6)   # 纵向拉长=发丝方向感
    # 红棕色系(与左侧红棕发统一)
    r_ = int(72 + rng8.uniform(0,52))    # R: 72-124
    g_ = int(42 + rng8.uniform(0,30))    # G: 42-72
    b_ = int(26 + rng8.uniform(0,20))    # B: 26-46
    dd.ellipse((px_-rx, py_-ry, px_+rx, py_+ry), fill=(r_, g_, b_))
img = img.filter(ImageFilter.GaussianBlur(0.5))
try: font=ImageFont.truetype("arial.ttf",20)
except: font=ImageFont.load_default()
ImageDraw.Draw(img).text((20,20), "VIRTUAL SELF v15 (ID-conditioned)", fill=(240,240,240), font=font)
img.save(r"B:\mode\vton_training\garment_poc\virtual_self_v15.png")
print("V15_DONE", flush=True)
