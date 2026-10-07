# 颜色标注升级:精确主色提取(K-means主色+色名映射)
import glob, os, json
import numpy as np
from PIL import Image

CN_COLORS = {"黑":(20,20,20),"白":(240,240,240),"灰":(128,128,128),"红":(200,40,40),"蓝":(50,80,180),
             "绿":(60,140,60),"黄":(220,200,60),"棕":(140,95,55),"粉":(230,160,170),"紫":(140,70,160),
             "橙":(230,140,50),"米":(225,215,190),"藏青":(35,45,85),"卡其":(190,175,135)}

def dominant_color(img, k=3):
    """K-means找主色簇(排除背景边缘)"""
    im = img.convert("RGB").resize((100, 100))
    W, H = im.size
    center = np.asarray(im)[int(H*0.2):int(H*0.8), int(W*0.2):int(W*0.8)].reshape(-1, 3).astype(np.float32)
    # 简易k-means
    rng = np.random.default_rng(0)
    cents = center[rng.choice(len(center), k, replace=False)]
    for _ in range(8):
        d = ((center[:, None]-cents[None])**2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            sel = center[lab==j]
            if len(sel): cents[j] = sel.mean(0)
    weights = np.bincount(lab, minlength=k)/len(lab)
    return cents[weights.argmax()]

def color_name(rgb):
    best, bd = None, 1e9
    for n, c in CN_COLORS.items():
        d = sum((a-b)**2 for a, b in zip(rgb, c))
        if d < bd: bd, best = d, n
    return best

out = {}
for f in glob.glob(r"B:\mode\brand_site\*\img_*.jpg") + glob.glob(r"B:\mode\brand_site\*\img_*.png") + glob.glob(r"B:\mode\brand_site\*\img_*.webp"):
    try:
        dom = dominant_color(Image.open(f))
        out[f] = color_name(dom)
    except Exception:
        pass
json.dump(out, open(r"B:\mode\style_train\color_labels_v2.json", "w", encoding="utf-8"), ensure_ascii=False)
print("COLOR_LABELS_DONE", len(out))
