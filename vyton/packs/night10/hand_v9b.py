
# 层3重做v9b:纹理源改为"手部特写图"(不是穿搭背景图)+更低强度+只注入手部
import numpy as np, glob, random
from PIL import Image
import cv2

BASE = r"B:\mode\vton_training"
img = np.asarray(Image.open(BASE + r"\e2e_hand_v8.png").convert("RGB")).astype(np.float32)
H, W = img.shape[:2]

# 纹理源:从v8图自己取"真实细节区"(戒指/指甲周边皮肤)自相似增强——零外部污染
# 法:高频自注入(own high-freq amplified)
lp = cv2.GaussianBlur(img, (0,0), 6)
hp = img - lp
# 皮肤mask(手部)
r, g, b = img[...,0], img[...,1], img[...,2]
skin = ((r>95)&(g>40)&(b>20)&(r>g)&(r>b)).astype(np.float32)
skin = cv2.GaussianBlur(skin, (0,0), 8)[..., None]
# 自高频放大1.6x注入(手部毛孔/纹理强化)
out = np.clip(lp + hp*1.6*skin + hp*1.0*(1-skin), 0, 255).astype(np.uint8)
# 轻锐化
sharp = cv2.detailEnhance(out, sigma_s=6, sigma_r=0.12)
out = np.clip(out.astype(np.float32)*0.7 + sharp.astype(np.float32)*0.3, 0, 255).astype(np.uint8)
Image.fromarray(out).save(BASE + r"\e2e_hand_v9b.png")
print("V9B_DONE 自高频注入")
