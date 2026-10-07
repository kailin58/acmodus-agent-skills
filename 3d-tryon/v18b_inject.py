# v18b:人脸框引导毛孔注入(blaze_face级简单方案:上半区肤色椭圆)
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

def face_region_pores(pil_img, seed=7, strength=1.2):
    arr = np.asarray(pil_img.convert("RGB")).copy().astype(np.float32)
    H, W = arr.shape[:2]
    # 简易脸区:上64%高度×中72%宽椭圆(与训练cond区一致)
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H*0.32, W*0.5
    ey, ex = H*0.30, W*0.36
    face = (((yy-cy)/ey)**2 + ((xx-cx)/ex)**2) < 1.0
    # 肤色复验(排除眉嘴眼)
    hsv = np.asarray(pil_img.convert("HSV"))
    h, s, v = hsv[...,0].astype(np.int32), hsv[...,1].astype(np.int32), hsv[...,2].astype(np.int32)
    skin = ((h < 28) | (h > 235)) & (s > 15) & (s < 200) & (v > 50)
    m = face & skin
    print(f"脸椭圆×肤色mask: {m.sum()} px ({m.mean()*100:.1f}%)")
    rng = np.random.default_rng(seed)
    dots = np.zeros((H, W))
    n = int(m.sum() * 0.35)  # mask内35%密度(加密)
    ys, xs = np.where(m)
    idx = rng.choice(len(ys), n, replace=False)
    dots[ys[idx], xs[idx]] = rng.choice([-1.0, 1.0], n) * rng.uniform(0.7, 1.0, n)
    pore = ndimage.gaussian_filter(dots, 0.9) * 30.0
    mott = ndimage.gaussian_filter(rng.normal(0,1,(H,W)), 8.0) * 1.6
    tex = (pore + mott) * strength
    out = arr + tex[..., None] * m[..., None]
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8))

base = Image.open(r"B:\mode\vton_training\student_v17_test.png")
W, H = base.size
mid = base.crop((W//3, 0, 2*W//3, H))
out = face_region_pores(mid, seed=11)
out.save(r"B:\mode\vton_training\student_v18b_test.png")
# 数字验证
from scipy import ndimage as nd
def tex_var(img):
    g = np.asarray(img.convert("L")).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    face = (((yy-H*0.32)/(H*0.30))**2 + ((xx-W*0.5)/(W*0.36))**2) < 1.0
    return nd.laplace(g)[face].var()
print(f"脸区纹理方差: v17={tex_var(mid):.1f} → v18b={tex_var(out):.1f}")
