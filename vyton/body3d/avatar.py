"""vyton.body3d.avatar — 详细数据→虚拟自己(高内聚:只管生成)
输入:age/sex/height/weight/ethnic → 输出:13718x3网格+meta
数据驱动:anthro_stages.json(WHO各年龄比例)+ethnic_adj.json(地区身高偏移)
"""
import numpy as np, json, os

_DIR = os.path.dirname(__file__)
STAGES = json.load(open(os.path.join(_DIR, "anthro_stages.json"), encoding="utf-8"))
ETHNIC = json.load(open(os.path.join(_DIR, "ethnic_adj.json"), encoding="utf-8"))

def stage_of(age, sex):
    if age < 1: return "baby_0"
    if age < 2: return "baby_1"
    if age < 4: return "child_3"
    if age < 7: return "child_6"
    if age < 10: return "child_9"
    if age < 13: return "teen_12"
    if age < 16: return "teen_15"
    if age < 65: return "adult_m" if sex == "m" else "adult_f"
    return "senior"

def gen_avatar(base_v, age, sex, height=None, weight=None, ethnic="东亚", region=None):
    st = stage_of(age, sex)
    cfg = dict(STAGES[st])
    adj = 0
    for k, v_ in ETHNIC.items():
        if k in ethnic: adj = v_; break
    h = height if height else cfg["h"] + adj
    h_m = h / 100
    if weight:
        bmi = weight / (h_m ** 2)
        ref = 22 if age >= 17 else 16
        lat_f = np.clip(1 + (bmi - ref) * 0.022, 0.78, 1.35)
    else:
        lat_f = 1.0
    v = base_v.copy()
    z = v[:, 2]; zmin, zmax = z.min(), z.max(); H = zmax - zmin
    hn = (z - zmin) / H
    newh = np.where(hn < 0.47, hn / 0.47 * cfg["leg"],
             np.where(hn < 0.87, cfg["leg"] + (hn - 0.47) / 0.40 * (1 - cfg["leg"] - cfg["head"]),
                      1 - cfg["head"] + (hn - 0.87) / 0.13 * cfg["head"]))
    scale = h / 100 / H
    out = v.copy()
    out[:, 2] = zmin + newh * H
    sh_f = cfg["shoulder"] / 0.25
    if sex == "f" and age >= 13: sh_f *= 0.95
    sh_w = np.clip((newh - 0.72) / 0.1, 0, 1) * np.clip((0.95 - newh) / 0.06, 0, 1)
    out[:, 0] = v[:, 0] * (1 + (sh_f - 1) * sh_w)
    if lat_f != 1.0:
        belly = np.exp(-((newh - 0.60) ** 2) / 0.008)
        chest = np.exp(-((newh - 0.75) ** 2) / 0.01) * 0.7
        limbs = np.where(newh < 0.55, 0.5, 0.25)
        headw = np.where(newh > 0.88, 0.05, 1.0)
        wgain = (belly + chest + limbs) * headw
        xf = 1 + (lat_f - 1) * np.clip(wgain, 0.05, 1.6) / 1.6
        yf = 1 + (lat_f - 1) * np.clip(wgain, 0.05, 1.3) / 1.4
        out[:, 0] = out[:, 0] * xf
        out[:, 1] = out[:, 1] * yf
    out = out * scale
    out[:, 2] -= out[:, 2].min()
    meta = {"stage": st, "age": age, "sex": sex, "height_cm": h,
            "weight_kg": weight, "ethnic": ethnic,
            "bmi": round(weight / h_m**2, 1) if weight else None}
    return out.astype(np.float32), meta
