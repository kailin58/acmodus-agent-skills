# Vyton-Measure v6:合成关键点路线(4万级×零噪声)
# Anny参数→mesh→采样33关键点(kps)→(kps,6维)训练对
import os, json
import numpy as np
import torch
import anny

OUT = r"B:\mode\measure\syn_kps"
os.makedirs(OUT, exist_ok=True)
model = anny.Anny(local_changes="default")
rng = np.random.default_rng(2026)
N = 40000
# 关键点索引模板:从第一个mesh的13718顶点里选33个(近似MediaPipe拓扑:头/肩/肘/腕/髋/膝/踝等)
# 用高度分位+横向极值自动选(无手标)
data = []
def pick_kps(V):
    z = V[:,2]; x = V[:,0]
    H = z.max()-z.min()
    def band(lo, hi):
        m = (z >= z.min()+H*lo) & (z <= z.min()+H*hi)
        return V[m]
    kps = []
    # 头顶0.99带宽/肩0.82/肘0.62/腕0.5/髋0.5/膝0.28/踝0.04
    for lo, hi, side in [(0.96,1.0,0),(0.80,0.85,-1),(0.80,0.85,1),(0.58,0.64,-1),(0.58,0.64,1),
                          (0.46,0.52,-1),(0.46,0.52,1),(0.44,0.48,0),(0.26,0.30,-1),(0.26,0.30,1),
                          (0.02,0.06,-1),(0.02,0.06,1)]:
        b = band(lo,hi)
        if len(b)==0: b = V
        if side==0: kps.append(b.mean(0))
        else:
            xs = b[:,0]
            kps.append(b[xs<=np.percentile(xs,15)].mean(0) if side<0 else b[xs>=np.percentile(xs,85)].mean(0))
    # 补到33:加左右眼/耳近似(头部细带)
    head = band(0.90,0.96)
    if len(head)>4:
        for p in [15,50,85]:
            xs = np.percentile(head[:,0], p)
            sel = head[np.abs(head[:,0]-xs)<0.02]
            kps.append(sel.mean(0) if len(sel) else head.mean(0))
    while len(kps)<33: kps.append(kps[-1])
    return np.array(kps[:33])

for i in range(N):
    h = float(rng.beta(2.2, 2.2)); w = float(rng.beta(2.0, 2.4))
    g = float(rng.choice([0.1, 0.9], p=[0.5, 0.5]))
    mu = float(rng.uniform(0.2, 0.8)); ag = float(rng.uniform(0.15, 0.85))
    pr = float(np.clip(rng.normal(0.5, 0.18), 0.05, 0.95))
    kw = {"height": h, "weight": w, "gender": g, "muscle": mu, "age": ag, "proportions": pr}
    try:
        V = np.asarray(model(None, kw)["vertices"])[0]
        kps = pick_kps(V)
        data.append({"kps": kps.astype(np.float32).round(4).tolist(), **kw})
    except Exception:
        continue
    if i % 4000 == 0:
        print(f"{i}/{N}", flush=True)
        json.dump(data, open(os.path.join(OUT, "data.json"), "w"))
json.dump(data, open(os.path.join(OUT, "data.json"), "w"))
print("SYN_KPS_DONE", len(data), flush=True)
