
# 虚拟自己v4:抖动采样+软边高斯点
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

face_img = Image.open(r"B:\mode\vton_training\face_test.png").convert("RGB")
face_arr = np.asarray(face_img)
fw, fh2 = face_img.size
lm_norm = np.load(r"B:\mode\vton_training\face_lm.npy")
flm = np.array([[p[0]*fw, p[1]*fh2] for p in lm_norm])
fx0, fx1 = flm[:,0].min(), flm[:,0].max()
fy0, fy1 = flm[:,1].min(), flm[:,1].max()

gi, gj = 200, 200
rng = np.random.default_rng(42)
W, H2 = 560, 600
img = Image.new("RGB", (W,H2), (20,20,24))
d = ImageDraw.Draw(img)
jit = (W-100)/gi * 0.85  # 抖动幅度≈0.85格距
for j in range(gj):
    for i in range(gi):
        uu, vv = i/(gi-1), j/(gj-1)
        px_ = 50 + uu*(W-100) + rng.uniform(-jit, jit)
        py_ = 60 + vv*(H2-120) + rng.uniform(-jit, jit)
        sx = int(np.clip(uu*(fx1-fx0)+fx0, 0, fw-1))
        sy = int(np.clip(vv*(fy1-fy0)+fy0, 0, fh2-1))
        r_, g_, b_ = face_arr[sy, sx, :3]
        edge = max(abs(uu-0.5), abs(vv-0.5))*2
        f = max(0.08, 1.0 - edge*edge*0.8)
        d.ellipse((px_-1.5, py_-1.5, px_+1.5, py_+1.5), fill=(int(r_*f),int(g_*f),int(b_*f)))
# 软边:轻微高斯(半径1)
img = img.filter(ImageFilter.GaussianBlur(0.8))
try: font=ImageFont.truetype("arial.ttf",20)
except: font=ImageFont.load_default()
d2 = ImageDraw.Draw(img)
d2.text((20,20), "VIRTUAL SELF v4 (jitter+soft)", fill=(240,240,240), font=font)
img.save(r"B:\mode\vton_training\garment_poc\virtual_self_v4.png")
print("V4_DONE", flush=True)
