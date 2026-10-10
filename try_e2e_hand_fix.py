
# Try e2e手部修复v1:检测手部区域→局部增强(超分+对比)
import sys, os, glob
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# MediaPipe hands(有模型?)
mp_hands = os.path.exists(r"B:\mode\models\hand_landmarker.task")
print("hand模型:", mp_hands)

# 轻量路线(不用检测器):极值启发——e2e图边缘区域的手(举起的手在上方边缘)
img = Image.open(r"B:\mode\vton_training\e2e_out_0.png").convert("RGB")
W, H = img.size
a = np.asarray(img).astype(np.float32)
# 皮肤色mask(YCbCr)
r, g, b = a[...,0], a[...,1], a[...,2]
cb = 128 - 0.168736*r - 0.331264*g + 0.5*b
cr = 128 + 0.5*r - 0.418688*g - 0.081312*b
skin = ((cb>77)&(cb<127)&(cr>133)&(cr<173))
# 手部=皮肤mask在边缘区(上1/4+左右边缘)
edge = np.zeros((H,W), bool)
edge[:H//4] = True; edge[:, :W//6] = True; edge[:, -W//6:] = True
hand = skin & edge
print(f"手部候选像素: {hand.sum()}")
if hand.sum() > 200:
    ys, xs = np.where(hand)
    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
    pad = 20
    box = (max(0,x1-pad), max(0,y1-pad), min(W,x2+pad), min(H,y2+pad))
    crop = img.crop(box)
    # 增强:2x超分(LANCZOS)+锐化+对比
    w, h = crop.size
    up = crop.resize((w*2, h*2), Image.LANCZOS)
    up = up.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
    up = ImageEnhance.Contrast(up).enhance(1.08)
    fin = up.resize((w, h), Image.LANCZOS)
    img.paste(fin, box[:2])
    img.save(r"B:\mode\vton_training\e2e_hand_fixed.png")
    print("手部增强完成:", box)
else:
    print("未检出明显手部区域")
