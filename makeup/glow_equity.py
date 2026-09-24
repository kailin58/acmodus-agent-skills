
# Vyton-Glow多族裔验证:7种族脸×统一妆容(检查各肤色下的效果一致性)
import io, os, json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from facemesh_lib.tasks import python as mp_python
from facemesh_lib.tasks.python import vision as mp_vision
from facemesh_lib import Image as MPImage, ImageFormat

OUT = r"B:\mode\glow_gallery"
df = pd.read_parquet(r"B:\mode\face_data\fairface_val.parquet", columns=["image","race"])
RACE_NAMES = {0:"White",1:"Black",2:"Latino",3:"EastAsian",4:"SEAsian",5:"Indian",6:"MidEast"}

det = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=r"B:\mode\models\face_landmarker.task"), num_faces=1))

LIP = [61,185,40,39,37,0,267,269,270,409,291,308,415,310,311,312,13,82,81,80,191]
EYE = [70,63,105,66,107,55,193,336,296,334,293,300]
CHEEK_L = [234,93,132,58,172,136,150,149,176,148]
CHEEK_R = [454,323,361,288,397,365,379,378,400,377]

ok, fail = 0, 0
results = {}
for race_code, rname in RACE_NAMES.items():
    sub = df[df["race"]==race_code].head(2)
    for i, row in sub.iterrows():
        try:
            b = row["image"].get("bytes")
            img = Image.open(io.BytesIO(b)).convert("RGB"); img.thumbnail((300,300))
            W, H = img.size
            arr = np.asarray(img).astype(np.float32)
            r0 = det.detect(MPImage(image_format=ImageFormat.SRGB, data=arr.astype(np.uint8)))
            if not r0.face_landmarks: fail+=1; continue
            lm = np.array([[p.x*W, p.y*H] for p in r0.face_landmarks[0]])
            def rmask(ids, blur):
                m = Image.new("L",(W,H),0)
                pts = np.array([lm[j] for j in ids], np.int32)
                ImageDraw.Draw(m).polygon(pts.reshape(-1).tolist(), fill=255)
                return np.asarray(m.filter(ImageFilter.GaussianBlur(blur))).astype(np.float32)/255.0
            out = arr.copy()
            def blend(mask, rgb, s):
                gray = out.mean(2, keepdims=True)/255.0
                tint = np.array(rgb,np.float32)[None,None,:]/255.0*(0.4+gray*1.2)
                out[:] = out*(1-mask[...,None]*s)+tint*255*mask[...,None]*s
            blend(rmask(LIP,3), (194,24,91), 0.7)
            blend(rmask(EYE,6), (196,168,141), 0.4)
            blend(rmask(CHEEK_L,12), (235,130,140), 0.3); blend(rmask(CHEEK_R,12), (235,130,140), 0.3)
            # 左右拼图(原图|上妆)
            after = Image.fromarray(np.clip(out,0,255).astype(np.uint8))
            combo = Image.new("RGB", (W*2, H), (14,14,18))
            combo.paste(Image.fromarray(arr.astype(np.uint8)), (0,0))
            combo.paste(after, (W,0))
            d = ImageDraw.Draw(combo)
            try: font = ImageFont.truetype("arial.ttf", 14)
            except: font = ImageFont.load_default()
            d.text((8,6), f"{rname}", fill=(255,255,255), font=font)
            combo.save(os.path.join(OUT, f"equity_{rname}_{ok}.png"))
            ok += 1
            results.setdefault(rname, 0); results[rname] += 1
        except Exception as e:
            fail += 1
json.dump({"ok": ok, "fail": fail, "by_race": results}, open(os.path.join(OUT, "equity_report.json"), "w"))
print("EQUITY_GLOW:", ok, "ok", fail, "fail", flush=True)
print(json.dumps(results), flush=True)
