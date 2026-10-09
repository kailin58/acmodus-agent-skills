# Glow合成配对:素颜图程序化上妆(唇/腮红/眼影区域 HSV调制)
import glob, os, random
import numpy as np
from PIL import Image

NM = glob.glob(r"B:\mode\glow_data\mt\extracted\non-makeup\**\*.jpg", recursive=True)
print(f"素颜源: {len(NM)}", flush=True)
os.makedirs(r"B:\mode\glow_data\pairs", exist_ok=True)

LIP_STYLES = [  # (色相偏移, 饱和增益, 亮度变化)
    (-8, 1.8, -10), (12, 2.2, -15), (0, 1.5, -8), (-15, 2.5, -20),  # 红/粉/自然/深红
]
BLUSH = [(5, 1.3, 8), (-5, 1.2, 6)]  # 腮红
EYE = [(-10, 1.4, -5), (8, 1.3, -8)]  # 眼影

def apply_makeup(img, rng):
    a = np.asarray(img.convert("HSV")).astype(np.float32)
    H, W = a.shape[:2]
    out = a.copy()
    # 唇区(下脸中部)
    lip = (slice(int(H*0.62), int(H*0.75)), slice(int(W*0.38), int(W*0.62)))
    dh, ds, dv = rng.choice(LIP_STYLES)
    region = out[lip]
    region[..., 0] = np.clip(region[..., 0] + dh, 0, 255)
    region[..., 1] = np.clip(region[..., 1] * ds, 0, 255)
    region[..., 2] = np.clip(region[..., 2] + dv, 0, 255)
    out[lip] = region
    # 腮红(两颊,软边界用距离衰减)
    dh, ds, dv = rng.choice(BLUSH)
    yy, xx = np.mgrid[0:H, 0:W]
    for cx in (0.28, 0.72):
        d = np.sqrt(((yy/H-0.48)*1.4)**2 + ((xx/W-cx)*2.2)**2)
        wgt = np.clip(1-d/0.35, 0, 1)
        out[..., 1] = np.clip(out[..., 1] * (1 + (ds-1)*wgt), 0, 255)
        out[..., 2] = np.clip(out[..., 2] + dv*wgt, 0, 255)
    # 眼影(眼周)
    dh, ds, dv = rng.choice(EYE)
    eye = (slice(int(H*0.35), int(H*0.50)), slice(int(W*0.25), int(W*0.75)))
    r2 = out[eye]
    r2[..., 0] = np.clip(r2[..., 0] + dh, 0, 255)
    r2[..., 1] = np.clip(r2[..., 1] * ds, 0, 255)
    out[eye] = r2
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "HSV").convert("RGB")
    # v2: float32全程

rng = random.Random(42)
n = 0
for i, p in enumerate(NM):
    try:
        src = Image.open(p).convert("RGB").resize((192, 192))
        for k in range(2):  # 每素颜2种妆=数据翻倍
            made = apply_makeup(src, rng)
            base = rf"B:\mode\glow_data\pairs\p{i:04d}_{k}"
            src.save(base + "_a.jpg", quality=92)
            made.save(base + "_b.jpg", quality=92)
            n += 2
    except Exception as e:
        import traceback
        print("ERR", i, str(e)[:60], flush=True)
        if i == 0:
            traceback.print_exc()
print(f"PAIRS_DONE {n//2}对({n}图)", flush=True)