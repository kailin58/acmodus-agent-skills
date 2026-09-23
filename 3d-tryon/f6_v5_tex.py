
# 虚拟自己v5:点颜色采样+亮度调制半径(真正的点彩构成)
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
img = Image.new("RGB", (W,H2), (20,20,24))
d = ImageDraw.Draw(img)
rng = np.random.default_rng(42)
# 4x密度(400x400=16万次尝试,按亮度概率保留→约6万点,暗区密亮区稀)
for j in range(400):
    for i in range(400):
        uu, vv = i/399, j/399
        sx = int(np.clip(uu*(fx1-fx0)+fx0, 0, fw-1))
        sy = int(np.clip(vv*(fy1-fy0)+fy0, 0, fh2-1))
        r_, g_, b_ = face_arr[sy, sx, :3]
        lum = (r_+g_+b_)/765
        edge = max(abs(uu-0.5), abs(vv-0.5))*2
        edge_f = max(0.05, 1.0 - edge*edge*0.8)
        # stippling:暗区保留概率高
        keep_p = (1.0 - lum*0.75) * edge_f
        if rng.random() > keep_p: continue
        px_ = 50 + uu*(W-100) + rng.uniform(-1.2, 1.2)
        py_ = 60 + vv*(H2-120) + rng.uniform(-1.2, 1.2)
        # 颜色=采样色×光照,半径=暗区大
        rad = 1.2 + (1.0-lum)*1.8
        d.ellipse((px_-rad, py_-rad, px_+rad, py_+rad),
                  fill=(int(r_*edge_f), int(g_*edge_f), int(b_*edge_f)))
img = img.filter(ImageFilter.GaussianBlur(0.6))
try: font=ImageFont.truetype("arial.ttf",20)
except: font=ImageFont.load_default()
d2 = ImageDraw.Draw(img)
d2.text((20,20), "VIRTUAL SELF v5 (stippling)", fill=(240,240,240), font=font)
img.save(r"B:\mode\vton_training\garment_poc\virtual_self_v5.png")
print("V5_DONE", flush=True)
