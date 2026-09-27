
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
# v11:头顶铺连续红棕底层(真体量,粒子只做边缘侵蚀)
ov = Image.new("RGBA", img.size, (0,0,0,0))
od = ImageDraw.Draw(ov)
W11, H11 = img.size
# 头顶弧形区(中央上1/3)
od.ellipse((W11*0.13, H11*0.02, W11*0.87, H11*0.46), fill=(96, 54, 32, 118))
ov = ov.filter(ImageFilter.GaussianBlur(9))
img = PImage.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
# v11:成束曲线丝(每束3-8根,沿头型向外下弯)
dd = ImageDraw.Draw(img)
rng8 = np.random.default_rng(7)
H8b = img.size[1]
for t in range(26):   # 束数
    uu = rng8.uniform(0.12, 0.88)
    px0 = 30 + uu*(img.size[0]-60)
    py0 = 40 + rng8.uniform(0, H8b*0.10)
    nb = rng8.integers(3, 9)        # 每束3-8根
    drift = rng8.uniform(-1, 1)     # 束整体弯向
    for k in range(nb):
        px_ = px0 + rng8.uniform(-3.5, 3.5)
        vv = rng8.uniform(0.04, 0.30)
    rx = 0.8 + rng8.uniform(0,0.6)
    ry = 3.0 + rng8.uniform(0,3.5)   # 更长竖丝
    u_ = rng8.uniform(0,1)
    if u_ < 0.55:      # 55%红棕主层(亮)
        r_, g_, b_ = int(105+rng8.uniform(0,50)), int(58+rng8.uniform(0,26)), int(32+rng8.uniform(0,16))
    elif u_ < 0.80:    # 25%深棕底层阴影
        r_, g_, b_ = int(62+rng8.uniform(0,26)), int(36+rng8.uniform(0,16)), int(22+rng8.uniform(0,10))
    else:              # 20%铜橙高光丝
        r_, g_, b_ = int(160+rng8.uniform(0,45)), int(96+rng8.uniform(0,30)), int(44+rng8.uniform(0,18))
    segs = 3
    for sg in range(segs):
        t_ = sg/(segs-1+1e-6)
        cx = px_ + drift*(vv*46)*t_ + rng8.uniform(-1,1)
        cy = 30 + vv*(H8b-100) + sg*(ry*0.9)
        dd.line([(cx, cy), (cx+drift*2.2, cy+ry*0.95)], fill=(r_, g_, b_), width=max(1,int(rx)))
img = img.filter(ImageFilter.GaussianBlur(0.5))
try: font=ImageFont.truetype("arial.ttf",20)
except: font=ImageFont.load_default()
ImageDraw.Draw(img).text((20,20), "VIRTUAL SELF v11 (auburn mass)", fill=(240,240,240), font=font)
img.save(r"B:\mode\vton_training\garment_poc\virtual_self_v11.png")
print("V11_DONE", flush=True)
