import json, os
import pyarrow.parquet as pq
pf = pq.ParquetFile(r"B:\mode\vton_training\data\VITON-HD-Captions\data\train-00000-of-00012.parquet")
t = pf.read()
caps = {}
for i in range(min(2200, t.num_rows)):
    txt = str(t.column("text")[i].as_py()).lower()
    caps[f"v{i:05d}.jpg"] = txt[:150]
json.dump(caps, open(r"B:\mode\style_train\data_v4\caps.json", "w", encoding="utf-8"), ensure_ascii=False)
print("caps:", len(caps))
