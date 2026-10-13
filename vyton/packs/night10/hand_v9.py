# 层3·纹理:频域换肤(治塑料感皮肤)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from PIL import Image
import cv2

BASE = r"B:\mode\vton_training"
img = np.asarray(Image.open(BASE + r"\e2e_hand_v8.png").convert("RGB")).astype(np.float32)
H, W = img.shape[:2]

# 皮肤纹理源:从同库真实人物照提高频(手部/皮肤区)
import glob, random
random.seed(9)
src_pool = sorted(glob.glob(r"B:\mode\pinterest\outfit_imgs\*.jpg"))
# 简易皮肤检测找含皮肤图
def skin_ratio(im):
    r, g, b = im[...,0], im[...,1], im[...,2]
    return float(((r>95)&(g>40)&(b>20)&(r>g)&(r>b)&(np.abs(r-g)>15)).mean())
tex_src = None
for p in random.sample(src_pool, 30):
    im = np.asarray(Image.open(p).convert("RGB").resize((1024,1024))).astype(np.float32)
    if skin_ratio(im) > 0.08:
        tex_src = im
        print("纹理源:", p.split("\\")[-1], flush=True)
        break
assert tex_src is not None, "无纹理源"

# 频域:img低频+tex_src高频
img_lp = cv2.GaussianBlur(img, (0,0), 8)
tex_hp = tex_src - cv2.GaussianBlur(tex_src, (0,0), 8)
# 皮肤mask(只对手/皮肤区注入)
r, g, b = img[...,0], img[...,1], img[...,2]
skin = ((r>95)&(g>40)&(b>20)&(r>g)&(r>b)).astype(np.float32)
skin = cv2.GaussianBlur(skin, (0,0), 12)[..., None]
blend = img_lp + tex_hp * 0.35 * skin
out = np.clip(blend, 0, 255).astype(np.uint8)
Image.fromarray(out).save(BASE + r"\e2e_hand_v9.png")
print("V9_TEX_DONE", flush=True)
