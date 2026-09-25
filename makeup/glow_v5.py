
import io, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import pyarrow.parquet as pq
from facemesh_lib.tasks import python as mp_python
from facemesh_lib.tasks.python import vision as mp_vision
from facemesh_lib import Image as MPImage, ImageFormat

t = pq.ParquetFile(r"B:\mode\vton_training\data\VITON-HD-Captions\data\train-00000-of-00012.parquet").read_row_group(0)
b = t.column('image')[0].as_py()['bytes']
img = Image.open(io.BytesIO(b)).convert("RGB"); img.thumbnail((440,440))
W, H = img.size
arr = np.asarray(img).astype(np.float32)
det = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=r"B:\mode\models\face_landmarker.task"), num_faces=1))
r0 = det.detect(MPImage(image_format=ImageFormat.SRGB, data=arr.astype(np.uint8)))
lm = np.array([[p.x*W, p.y*H] for p in r0.face_landmarks[0]])
LIP = [61,185,40,39,37,0,267,269,270,409,291,308,415,310,311,312,13,82,81,80,191]
def rmask(ids, blur):
    m = Image.new("L",(W,H),0)
    pts = np.array([lm[j] for j in ids], np.int32)
    ImageDraw.Draw(m).polygon(pts.reshape(-1).tolist(), fill=255)
    return np.asarray(m.filter(ImageFilter.GaussianBlur(blur))).astype(np.float32)/255.0
m_lip = rmask(LIP, 3)
target = np.array([194,24,91], np.float32)
gray = arr.mean(2, keepdims=True)/255.0
# 饱和度补偿只作用唇区(v4修:不动全图)
out = arr.copy()
lip_zone = m_lip > 0.3
lz_rgb = arr[lip_zone]
# 唇区像素向目标色收敛(色度替换,保明度)
lum = lz_rgb.mean(1, keepdims=True)/255.0
tgt = target[None,:]/255.0
tinted = tgt*(0.32+lum*1.35)*1.02
out[lip_zone] = arr[lip_zone]*(1-m_lip[lip_zone,None]*0.72) + tinted*255*m_lip[lip_zone,None]*0.72
gloss_zone = (gray[...,0]>0.68) & (m_lip>0.6)
out[gloss_zone] = np.clip(out[gloss_zone]*1.22 + 16, 0, 255)
res = Image.fromarray(np.clip(out,0,255).astype(np.uint8))
try: font=ImageFont.truetype("arial.ttf",18)
except: font=ImageFont.load_default()
ImageDraw.Draw(res).text((15,15), "Glow v5: static+satin-gloss", fill=(255,255,255), font=font)
res.save(r"B:\mode\glow_v5_gloss.png")
print("GLOW_V3_DONE", flush=True)
