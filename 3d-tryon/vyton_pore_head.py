# vyton_pore_head v1 — 运行时毛孔注入头(端侧3ms,CPU可跑)
# 用法: out = pore_head(student_output_pil)  # PIL in/out
import numpy as np
from PIL import Image
from scipy import ndimage as nd

def pore_head(pil_img, seed=11, density=0.5, amp=38.0, mott_a=2.0):
    arr = np.asarray(pil_img.convert("RGB")).copy().astype(np.float32)
    H, W = arr.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    face = (((yy-H*0.32)/(H*0.30))**2 + ((xx-W*0.5)/(W*0.36))**2) < 1.0
    hsv = np.asarray(pil_img.convert("HSV"))
    h, s, v = hsv[...,0].astype(np.int32), hsv[...,1].astype(np.int32), hsv[...,2].astype(np.int32)
    skin = ((h < 28) | (h > 235)) & (s > 15) & (s < 200) & (v > 50)
    m = face & skin
    rng = np.random.default_rng(seed)
    dots = np.zeros((H, W))
    n = int(m.sum() * density)
    ys, xs = np.where(m)
    idx = rng.choice(len(ys), n, replace=False)
    dots[ys[idx], xs[idx]] = rng.choice([-1.0, 1.0], n) * rng.uniform(0.7, 1.0, n)
    pore = nd.gaussian_filter(dots, 0.9) * amp
    mott = nd.gaussian_filter(rng.normal(0,1,(H,W)), 8.0) * mott_a
    out = arr + (pore + mott)[..., None] * m[..., None]
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8))
