# 毛孔头v3:全流程优化(预计算mask+减PIL往返+缓存kernel)
import numpy as np
from PIL import Image
from scipy import ndimage as nd

_K = None
_MASK_CACHE = {}

def _kernel(sigma=0.9):
    global _K
    if _K is None:
        r = int(3*sigma)
        x = np.arange(-r, r+1)
        _K = np.exp(-x*x/(2*sigma*sigma)); _K /= _K.sum()
    return _K

def _mask(H, W):
    key = (H, W)
    if key not in _MASK_CACHE:
        yy, xx = np.mgrid[0:H, 0:W]
        _MASK_CACHE[key] = (((yy-H*0.32)/(H*0.30))**2 + ((xx-W*0.5)/(W*0.36))**2) < 1.0
    return _MASK_CACHE[key]

def pore_head_v3(pil_img, seed=11, density=0.5, amp=38.0, mott_a=2.0):
    arr = np.asarray(pil_img.convert("RGB"))
    H, W = arr.shape[:2]
    # HSV只取需要通道(减转换)
    r_, g_, b_ = arr[...,0].astype(np.int16), arr[...,1].astype(np.int16), arr[...,2].astype(np.int16)
    mx = np.maximum(np.maximum(r_, g_), b_)
    mn = np.minimum(np.minimum(r_, g_), b_)
    v_ = mx
    s_ = np.where(mx > 0, (mx-mn)*255//np.maximum(mx,1), 0)
    # hue近似不需要精确(皮肤判定用rgb关系)
    skin = ((r_ > g_) & (r_ > b_) & (s_ > 15) & (s_ < 200) & (v_ > 50) & (v_ < 250))
    m = _mask(H, W) & skin
    rng = np.random.default_rng(seed)
    dots = np.zeros((H, W))
    ys, xs = np.where(m)
    if len(ys) == 0:
        return pil_img
    n = int(len(ys) * density)
    idx = rng.choice(len(ys), n, replace=False)
    dots[ys[idx], xs[idx]] = rng.choice([-1.0, 1.0], n) * rng.uniform(0.7, 1.0, n)
    k = _kernel(0.9)
    pore = nd.convolve1d(nd.convolve1d(dots, k, axis=1), k, axis=0) * amp
    mott = nd.gaussian_filter(rng.normal(0,1,(H,W)), 8.0) * mott_a
    tex = (pore + mott) * m
    out = arr.astype(np.int16)
    for c in range(3):
        out[..., c] += tex.astype(np.int16)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

if __name__ == "__main__":
    import time
    img = Image.new("RGB", (256,256), (200,160,130))
    t0 = time.time()
    for _ in range(20):
        pore_head_v3(img, seed=11)
    print(f"v3: {(time.time()-t0)/20*1000:.2f}ms (v1=7.8ms)")
