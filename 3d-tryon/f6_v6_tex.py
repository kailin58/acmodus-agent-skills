
# 虚拟自己v6:去边缘伪影+中间调点接管(底层只留最亮高光)
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
        keep_p = min(1.0, 0.25 + (1.0-lum)*0.85)  # 中间调也接管(0.25保底)
        if rng.random() > keep_p * edge_f: continue
        px_ = 40 + uu*(W-80) + rng.uniform(-1.1, 1.1)
        py_ = 50 + vv*(H2-90) + rng.uniform(-1.1, 1.1)
        rad = 1.1 + (1.0-lum)*1.9
        d.ellipse((px_-rad, py_-rad, px_+rad, py_+rad),
                  fill=(int(r_*edge_f), int(g_*edge_f), int(b_*edge_f)))
img = img.filter(ImageFilter.GaussianBlur(0.5))
try: font=ImageFont.truetype("arial.ttf",20)
except: font=ImageFont.load_default()
ImageDraw.Draw(img).text((20,20), "VIRTUAL SELF v6 (midtone)", fill=(240,240,240), font=font)
img.save(r"B:\mode\vton_training\garment_poc\virtual_self_v6.png")
print("V6_DONE", flush=True)
