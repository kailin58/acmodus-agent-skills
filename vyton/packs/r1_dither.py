# VytonSR-R1:抖动预处理防banding(编码前注入)
import numpy as np
from PIL import Image
import subprocess, os

def ordered_dither(img, strength=6):
    """8x8 Bayer有序抖动:打散暗部banding"""
    a = img.astype(np.float32)
    H, W = a.shape[:2]
    bayer = np.array([[0,32,8,40,2,34,10,42],
                      [48,16,56,24,50,18,58,26],
                      [12,44,4,36,14,46,6,38],
                      [60,28,52,20,62,30,54,22],
                      [3,35,11,43,1,33,9,41],
                      [51,19,59,27,49,17,57,25],
                      [15,47,7,39,13,45,5,37],
                      [63,31,55,23,61,29,53,21]], np.float32) / 64.0 - 0.5
    tile = np.tile(bayer, (H//8+1, W//8+1))[:H, :1]
    # 只对暗部区注入(亮部banding少)
    lum = a.mean(-1, keepdims=True)
    dark = np.clip(1.0 - lum/128.0, 0, 1)
    d = np.concatenate([tile]*3, axis=-1)[:H, :W] if a.ndim == 3 else tile[:H, :W]
    out = a + d * strength * dark
    return np.clip(out, 0, 255).astype(np.uint8)

src = Image.open(r"B:/mode/tools/d2_test.png").convert("RGB")
dith = ordered_dither(np.asarray(src))
Image.fromarray(dith).save(r"B:/mode/tools/d2_dither.png")
print("R1_DITHER_OK")
