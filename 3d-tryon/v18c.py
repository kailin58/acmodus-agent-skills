# v18c:毛孔幅度强化版
import numpy as np
from PIL import Image
from scipy import ndimage as nd

def face_region_pores(pil_img, seed=7, density=0.5, amp=38.0, mott_a=2.0):
    arr = np.asarray(pil_img.convert("RGB")).copy().astype(np.float32)
    H, W = arr.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H*0.32, W*0.5
    face = (((yy-cy)/(H*0.30))**2 + ((xx-cx)/(W*0.36))**2) < 1.0
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
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8)), m

base = Image.open(r"B:\mode\vton_training\student_v17_test.png")
W0, H0 = base.size
mid = base.crop((W0//3, 0, 2*W0//3, H0))
out, m = face_region_pores(mid, seed=11)
out.save(r"B:\mode\vton_training\student_v18c_test.png")

def tex_var(img):
    g = np.asarray(img.convert("L")).astype(np.float32)
    return nd.laplace(g)[m].var()
v1, v2 = tex_var(mid), tex_var(out)
print(f"脸区纹理: v17={v1:.1f} → v18c={v2:.1f} ({(v2/v1-1)*100:+.1f}%)")
