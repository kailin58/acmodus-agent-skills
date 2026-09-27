# v6.1特征工程:17解剖点+8围度带(重生成4万)
import os, json
import numpy as np
import anny

OUT = r"B:\mode\measure\syn_kps_v3"
os.makedirs(OUT, exist_ok=True)
model = anny.Anny(local_changes="default")
rng = np.random.default_rng(31337)
N = 100000

def feats(V):
    z = V[:,2]; x = V[:,0]
    H = z.max()-z.min()
    def band_w(lo, hi):
        m = (z >= z.min()+H*lo) & (z <= z.min()+H*hi)
        if m.sum() < 4: return 0.0
        return float(x[m].max()-x[m].min())
    def band_c(lo, hi):
        m = (z >= z.min()+H*lo) & (z <= z.min()+H*hi)
        return V[m].mean(0) if m.sum() else V.mean(0)
    # 17解剖点:头顶/眼×2/颈/肩×2/肘×2/腕×2/胸/腰/髋×2/膝×2/踝×2
    pts = [band_c(0.955,1.0)]
    eye = band_c(0.90,0.95); pts += [eye, eye]
    pts.append(band_c(0.84,0.88))          # 颈
    shL = band_c(0.80,0.845); shR = shL     # 占位
    pts += [shL, shR]
    pts += [band_c(0.62,0.67)]*2            # 肘
    pts += [band_c(0.50,0.55)]*2            # 腕
    pts.append(band_c(0.52,0.58))           # 胸
    pts.append(band_c(0.40,0.46))           # 腰
    hip = band_c(0.30,0.36); pts += [hip, hip]
    pts += [band_c(0.13,0.18)]*2            # 膝
    pts += [band_c(0.01,0.05)]*2            # 踝
    kp = np.array(pts[:17], dtype=np.float32)
    # 8围度带:颈/胸/腰/髋/大腿L/大腿R/上臂L/上臂R
    widths = [band_w(0.845,0.875), band_w(0.52,0.58), band_w(0.40,0.46), band_w(0.30,0.36),
              band_w(0.13,0.18), band_w(0.13,0.18), band_w(0.60,0.68), band_w(0.60,0.68)]
    return kp, np.array(widths, dtype=np.float32)

data = []
for i in range(N):
    h = float(rng.beta(2.2, 2.2)); w = float(rng.beta(2.0, 2.4))
    g = float(rng.choice([0.1, 0.9], p=[0.5, 0.5]))
    mu = float(rng.uniform(0.2, 0.8)); ag = float(rng.uniform(0.15, 0.85))
    pr = float(np.clip(rng.normal(0.5, 0.18), 0.05, 0.95))
    kw = {"height": h, "weight": w, "gender": g, "muscle": mu, "age": ag, "proportions": pr}
    try:
        V = np.asarray(model(None, kw)["vertices"])[0]
        kp, wd = feats(V)
        data.append({"kp": kp.round(4).tolist(), "wd": wd.round(4).tolist(), **kw})
    except Exception:
        continue
    if i % 5000 == 0:
        print(f"{i}/{N}", flush=True)
        json.dump(data, open(os.path.join(OUT, "data.json"), "w"))
json.dump(data, open(os.path.join(OUT, "data.json"), "w"))
print("V61_GEN_DONE", len(data), flush=True)
