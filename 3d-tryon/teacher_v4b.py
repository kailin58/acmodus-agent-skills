# 教师v4b:毛孔加幅3倍+多尺度(细毛孔+中频皮色斑驳8-16px)——按验收整改
import glob, os
import numpy as np
from PIL import Image
from scipy import ndimage

def add_skin_texture_v2(img, strength=1.0):
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    hsv = np.asarray(img.convert("HSV"))
    h, s = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32)
    skin = ((h < 28) | (h > 235)) & (s > 20) & (s < 180)
    skin = ndimage.binary_dilation(skin, iterations=2)
    rng = np.random.default_rng(42)
    # 细毛孔(2-3px频率) σ=4/255×3倍幅度
    pore = rng.normal(0, 1, arr.shape[:2])
    pore = ndimage.gaussian_filter(pore, 1.0)
    # 中频皮色斑驳(8-16px)——"有皮肤感"的关键
    mottle = rng.normal(0, 1, arr.shape[:2])
    mottle = ndimage.gaussian_filter(mottle, 6.0)
    mottle2 = rng.normal(0, 1, arr.shape[:2])
    mottle2 = ndimage.gaussian_filter(mottle2, 12.0)
    tex = pore * 4.5 + mottle * 2.8 + mottle2 * 2.0
    tex = tex * strength
    m = skin[..., None].astype(np.float32)
    out = arr + tex[..., None] * m
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

srcs = sorted(glob.glob(r"B:\mode\cloud_out\tv3\tv3_*.png"))
os.makedirs(r"B:\mode\vton_training\teacher_v4b_pore", exist_ok=True)
for i, p in enumerate(srcs):
    im = Image.open(p)
    add_skin_texture_v2(im, 1.0).save(rf"B:\mode\vton_training\teacher_v4b_pore\tb_{i:04d}.png")
print("V4B_DONE", len(srcs), flush=True)
