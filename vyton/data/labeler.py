# YT数据标注器:词采集的缩略图→意图真值(复用Pinterest法)
import json, os, glob
IDX = r"B:\mode\yt_outfit\yt_index.json"
idx = json.load(open(IDX, encoding="utf-8"))

STYLE = ["minimalist","french","vintage","street","sporty","elegant","casual","boho"]
SCENE = ["work","date","party","casual","travel","home"]
S_MAP = {"korean":"street","japanese":"street","street":"street","vintage":"vintage",
         "minimalist":"minimalist","capsule":"minimalist","glam":"elegant","natural":"minimalist"}
C_MAP = {"office":"work","work":"work","date":"date","winter":"casual","summer":"casual",
         "daily":"casual","street":"casual","party":"party"}

out = {}
for kw, d in idx.items():
    st = list({S_MAP[k] for k in S_MAP if k in kw.lower()})[:2]
    sc = list({C_MAP[k] for k in C_MAP if k in kw.lower()})[:2]
    for vid in d.get("videos", []):
        f = rf"B:\mode\yt_outfit\{vid}.jpg"
        if os.path.exists(f) and os.path.getsize(f) > 5000:
            out[f] = {"style": st, "scene": sc, "src": "yt", "kw": kw}
json.dump(out, open(r"B:\mode\style_train\yt_labels.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"YT标注: {len(out)}张")