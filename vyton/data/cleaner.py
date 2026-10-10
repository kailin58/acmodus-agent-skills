# v10数据清洗:颜色直方图+边缘密度过滤非穿搭图(纯文字/风景/表情包)
import glob, os, json
import numpy as np
from PIL import Image, ImageFilter

def is_fashion(img):
    """穿搭图启发式:色彩多样+有人物结构(竖构图或方图)+内容丰富"""
    im = img.convert("RGB").resize((128,128))
    a = np.asarray(im)
    # 1)色彩:穿搭图色相分布宽
    hsv = np.asarray(im.convert("HSV"))
    h_std = hsv[...,0].std()
    # 2)细节:衣服纹理/人物轮廓=中等边缘密度
    e = np.asarray(im.convert("L").filter(ImageFilter.FIND_EDGES)).astype(np.float32)
    e_dens = (e > 40).mean()
    # 3)非纯色/非文字(文字=极高对比稀疏)
    lum_std = a.mean(axis=2).std()
    ok = (h_std > 18) and (0.03 < e_dens < 0.45) and (lum_std > 25)
    return ok, (round(h_std,1), round(e_dens,3), round(lum_std,1))

labs = json.load(open(r"B:\mode\style_train\pin_labels_v1.json", encoding="utf-8"))
keep, drop = {}, {}
for f, lab in labs.items():
    if not os.path.exists(f): continue
    try:
        ok, stats = is_fashion(Image.open(f))
        (keep if ok else drop)[f] = stats
    except Exception: drop[f] = "err"
json.dump(list(keep.keys()), open(r"B:\mode\style_train\pin_clean_v1.json","w"))
print(f"清洗: 留{len(keep)} 弃{len(drop)} ({len(keep)/(len(keep)+len(drop))*100:.0f}%保留)")
# 弃样例统计
import collections
ds = [v[2] if isinstance(v,tuple) else 0 for v in drop.values()]
print("弃图lum_std分布(纯文字图<15):", sorted(ds)[:5])