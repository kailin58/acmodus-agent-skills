# Style v4正式数据:VITON 2000张caption弱标签品类
import io, os, json
import pyarrow.parquet as pq
from PIL import Image

OUT = r"B:\mode\style_train\data_v4"
os.makedirs(OUT, exist_ok=True)
CATS = ["dress","top","shirt","pants","skirt","coat","jacket","shorts","sweater"]
data = []
pf = pq.ParquetFile(r"B:\mode\vton_training\data\VITON-HD-Captions\data\train-00000-of-00012.parquet")
t = pf.read()  # 全分片
n = 0
for i in range(min(2200, t.num_rows)):
    try:
        row = {c: t.column(c)[i].as_py() for c in t.column_names}
        img_b = row.get("image") or row.get("image_url") or list(row.values())[0]
        if isinstance(img_b, dict): img_b = img_b.get("bytes")
        cap = str(row.get("text", "")).lower()
        cats = [c for c in CATS if c in cap]
        if not cats: continue
        img = Image.open(io.BytesIO(img_b)).convert("RGB")
        img.thumbnail((224,224))
        fn = f"v{i:05d}.jpg"
        img.save(os.path.join(OUT, fn), quality=88)
        data.append({"img": fn, "cats": cats, "colors": []})
        n += 1
        if n >= 2000: break
    except Exception:
        continue
json.dump(data, open(os.path.join(OUT, "labels.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("STYLE_V4_DATA", n)
