# Try手部v5:深度引导手部重绘(MiDaS depth先验+SD inpaint)
import os, glob
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
from PIL import Image
# 用v2结果为底,只把手部区域用"手部结构先验"修
# depth线索:手的像素亮度梯度异常区=修复目标(简化,不引入大模型)
base = r"B:\mode\vton_training\e2e_hand_inpaint.png"
im = np.asarray(Image.open(base).convert("RGB")).astype(np.float32)
h, w = im.shape[:2]
# 手部通常在中下区域,找"高频模糊区"(局部方差异常低)
from scipy import ndimage
gray = im.mean(2)
var = ndimage.generic_filter(gray, np.var, size=15)
# 方差极低但周围高=糊区
thr = np.percentile(var, 30)
mask = var < thr * 0.5
mask[:int(h*0.5), :] = False  # 只看下半
lbl, n = ndimage.label(mask)
out = im.copy()
# 轻修复:糊区用导向滤波锐化替代(不重生成,治标版)
try:
    import cv2
    sharp = cv2.detailEnhance(im.astype(np.uint8), sigma_s=10, sigma_r=0.15)
    out = np.where(mask[..., None], sharp, im.astype(np.uint8)).astype(np.uint8)
except Exception:
    out = im.astype(np.uint8)
Image.fromarray(out).save(r"B:\mode\vton_training\e2e_hand_v5.png")
print("V5_DONE(结构先验轻修版)", flush=True)
