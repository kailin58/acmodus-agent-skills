# 层4·光影:手部光影一致性(亮度统计匹配)
import numpy as np
from PIL import Image
import cv2

BASE = r"B:\mode\vton_training"
img = np.asarray(Image.open(BASE + r"\e2e_hand_v9.png").convert("RGB")).astype(np.float32)
H, W = img.shape[:2]
# 全图(衣区)亮度统计 vs 手部区
r, g, b = img[...,0], img[...,1], img[...,2]
skin = ((r>95)&(g>40)&(b>20)&(r>g)&(r>b))
hand_zone = np.zeros_like(skin)
hand_zone[:int(H*0.62), :] = True
hand = skin & hand_zone
cloth = ~skin
h_mean = img[hand].mean()
c_mean = img[cloth].mean()
print(f"手部亮度{h_mean:.0f} vs 衣区{c_mean:.0f}", flush=True)
# 目标:手部亮度对齐衣区光照的相对关系(略亮,自然光照手更受光)
gain = min(1.12, max(0.94, c_mean / max(h_mean,1) * 1.04))
out = img.copy()
out[..., :3][hand] = np.clip(img[..., :3][hand] * gain, 0, 255)
# 手部投影:手下方衣料加软阴影(光影联动)
ys, xs = np.where(hand)
if len(ys):
    y_bot = ys.max()
    shadow = np.zeros((H, W), np.float32)
    cv2.ellipse(shadow, (int(xs.mean()), min(H-1, y_bot+28)), (int(W*0.12), 18), 0, 0, 360, 0.35, -1)
    shadow3 = (1 - shadow)[..., None]
    clothm = (~skin).astype(np.float32)[..., None]
    out[..., :3] = np.clip(out[..., :3] * (1 - clothm*0.18*shadow3), 0, 255)
Image.fromarray(np.clip(out,0,255).astype(np.uint8)).save(BASE + r"\e2e_hand_v10.png")
print(f"V10_LIGHT_DONE gain={gain:.2f}", flush=True)
