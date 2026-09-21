
# 颜色搭配共现统计v0:53张搭配照→主色提取→共现矩阵
import os, json
import numpy as np
from PIL import Image
from collections import Counter

SRC = r"B:\mode\pinterest\images"
# 12色系(与风格识别器v0一致)
def rgb2name(r, g, b):
    mx, mn = max(r,g,b), min(r,g,b)
    v = (int(r)+int(g)+int(b))/3
    if v > 235 and mx-mn < 25: return "white"
    if v < 45: return "black"
    if mx-mn < 25: return "gray" if v < 170 else "light_gray"
    h = None
    import colorsys
    hh, ss, vv = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    if ss < 0.15: return "gray" if v < 170 else "light_gray"
    hdeg = hh * 360
    if hdeg < 15 or hdeg >= 345: return "red"
    if hdeg < 40: return "orange"
    if hdeg < 68: return "yellow"
    if hdeg < 160: return "green"
    if hdeg < 200: return "cyan"
    if hdeg < 260: return "blue"
    if hdeg < 300: return "purple"
    return "pink"

pairs = Counter()
singles = Counter()
n_img = 0
for f in os.listdir(SRC):
    if not f.lower().endswith((".jpg",".png",".webp")): continue
    try:
        img = Image.open(os.path.join(SRC, f)).convert("RGB").resize((96, 128))
        a = np.array(img).reshape(-1, 3)
        # 采样
        cnt = Counter(rgb2name(*px) for px in a[::7])
        # 主色=占比前3(排除背景嫌疑的白/灰过半时降权)
        top = [c for c, _ in cnt.most_common(6)]
        # 去背景色:占比最大的若超60%视为背景,剔除
        total = sum(cnt.values())
        main = [c for c, n in cnt.most_common(8) if n/total < 0.55][:3]
        # 两两共现
        for i in range(len(main)):
            singles[main[i]] += 1
            for j in range(i+1, len(main)):
                p = tuple(sorted((main[i], main[j])))
                pairs[p] += 1
        n_img += 1
    except Exception:
        pass

# 规则表:共现频次+共现率
rules = []
for (a, b), n in pairs.most_common(30):
    conf = n / max(singles[a], singles[b])
    rules.append({"colors": [a, b], "count": n, "confidence": round(conf, 2)})
json.dump({"images": n_img, "rules": rules}, 
          open(r"B:\mode\color_rules_v0.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"颜色规则v0: {n_img}张图, {len(rules)}条规则")
for r in rules[:8]:
    print(f"  {r['colors'][0]}+{r['colors'][1]}: {r['count']}次 置信{r['confidence']}")
