# 教师v4c:排发mask+毛孔加密+斑驳减半
import glob, os
import numpy as np
from PIL import Image
from scipy import ndimage

def skin_only_mask(img):
    hsv = np.asarray(img.convert("HSV"))
    h, s, v = hsv[...,0].astype(np.int32), hsv[...,1].astype(np.int32), hsv[...,2].astype(np.int32)
    arr = np.asarray(img.convert("L")).astype(np.float32)
    # 纹理能量:头发区高频能量高(发丝),皮肤低
    lap = np.abs(ndimage.laplace(ndimage.gaussian_filter(arr, 1.0)))
    smooth = lap < 14.0
    m = ((h < 28) | (h > 235)) & (s > 20) & (s < 180) & (v > 60) & (v < 250) & smooth
    return ndimage.binary_dilation(m, iterations=1)

def add_pores(img, strength=1.0):
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    skin = skin_only_mask(img)
    H, W = arr.shape[:2]
    rng = np.random.default_rng(7)
    # 离散毛孔:高斯点阵(2px核)——密度加密
    dots = np.zeros((H, W))
    ys, xs = rng.integers(0, H, 40000), rng.integers(0, W, 40000)
    vals = rng.choice([-1.0, 1.0], 40000) * rng.uniform(0.6, 1.0, 40000)
    dots[ys, xs] = vals
    pore = ndimage.gaussian_filter(dots, 1.0) * 26.0   # 细离散孔隙
    # 中频皮色斑驳(权重减半)
    mott = ndimage.gaussian_filter(rng.normal(0,1,(H,W)), 8.0) * 1.4
    tex = (pore + mott) * strength
    m = skin[..., None].astype(np.float32)
    out = arr + tex[..., None] * m
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

srcs = sorted(glob.glob(r"B:\mode\cloud_out\tv3\tv3_*.png"))
os.makedirs(r"B:\mode\vton_training\teacher_v4c_pore", exist_ok=True)
for i, p in enumerate(srcs):
    add_pores(Image.open(p), 1.0).save(rf"B:\mode\vton_training\teacher_v4c_pore\tc_{i:04d}.png")
print("V4C_DONE", len(srcs), flush=True)
