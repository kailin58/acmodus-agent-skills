# v5c:民族图补颜色弱标注(HSV直方图主色)
import glob, os
import numpy as np
from PIL import Image

def dominant_colors(img, topn=2):
    """HSV量化直方图→主色名"""
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    mask = (V > 40) & (V < 250)  # 排除近黑近白
    if mask.sum() < 500:
        return []
    # 显著色像素(饱和度够)
    m2 = mask & (S > 60)
    if m2.sum() < 300: m2 = mask
    hh = H[m2]
    names = []
    for h in [np.median(hh)] + ([] if topn==1 else [np.percentile(hh, 25), np.percentile(hh, 75)]):
        h8 = h / 255 * 360
        if h8 < 15 or h8 >= 345: n = "red"
        elif h8 < 45: n = "orange" if S[m2].mean() > 100 else "brown"
        elif h8 < 70: n = "yellow"
        elif h8 < 160: n = "green"
        elif h8 < 200: n = "blue"
        elif h8 < 260: n = "purple"
        elif h8 < 330: n = "pink"
        else: n = "red"
        if n not in names: names.append(n)
    # 亮暗补白黑
    if (V > 235).mean() > 0.3 and "white" not in names: names.insert(0, "white")
    if (V < 30).mean() > 0.3 and "black" not in names: names.insert(0, "black")
    return names[:topn+1]

if __name__ == "__main__":
    import json
    out = {}
    for cdir in sorted(glob.glob(r"B:\mode\ethnic_data\*")):
        c = os.path.basename(cdir)
        out[c] = {}
        for f in glob.glob(os.path.join(cdir, "*.jpg")):
            try:
                cs = dominant_colors(Image.open(f).resize((64,64)), 2)
                out[c][os.path.basename(f)] = cs
            except Exception:
                out[c][os.path.basename(f)] = []
    json.dump(out, open(r"B:\mode\ethnic_data\_colors.json", "w"), ensure_ascii=False, indent=1)
    tot = sum(1 for c in out.values() for v in c.values() if v)
    print("有颜色标签:", tot, "/", sum(len(c) for c in out.values()))
