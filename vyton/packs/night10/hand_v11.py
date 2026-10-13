# v11终极:甲饰边缘+手背次表面散射+全局噪声统一
import numpy as np
from PIL import Image
import cv2

BASE = "B:/mode/vton_training"
img = np.asarray(Image.open(BASE + "/e2e_hand_v10b.png").convert("RGB")).astype(np.float32)
H, W = img.shape[:2]
r, g, b = img[...,0], img[...,1], img[...,2]
skin_raw = ((r>95)&(g>40)&(b>20)&(r>g)&(r>b)).astype(np.float32)
skin = cv2.GaussianBlur(skin_raw, (0,0), 8)[..., None]

# 1.甲饰边缘锐化:高光区(指甲)局部CLAHE+unsharp
lum = img.mean(2, keepdims=True)
nail = (lum > np.percentile(lum, 88)).astype(np.float32)
nail = cv2.GaussianBlur(nail, (0,0), 3)[..., None]
sharp = img + (img - cv2.GaussianBlur(img, (0,0), 2.2)) * 0.6
img2 = img * (1 - nail*0.6) + sharp * (nail*0.6)

# 2.手背次表面散射:皮肤暖色低频渗透(红通道模糊混入)+高光蜡感柔化
ss = cv2.GaussianBlur(img2 * np.array([1.15, 0.9, 0.85]), (0,0), 14)
img3 = img2 * 0.88 + ss * 0.12 * skin
# 高光柔化:过亮皮肤压一点(去蜡平亮)
hi = (lum > 200).astype(np.float32)
hi = cv2.GaussianBlur(hi, (0,0), 6)[..., None]
img3 = img3 * (1 - hi*0.10*skin) + (img3*0.94) * (hi*0.10*skin)

# 3.全局噪声统一:全图加同参数细颗粒(皮肤与背景同源噪声)
rng = np.random.default_rng(11)
uni = cv2.GaussianBlur(rng.normal(0, 1.4, (H, W, 1)).astype(np.float32), (0,0), 0.8)[..., None]
img4 = img3 + uni * 0.8

out = np.clip(img4, 0, 255).astype(np.uint8)
Image.fromarray(out).save(BASE + "/e2e_hand_v11.png")
print("V11_DONE")
