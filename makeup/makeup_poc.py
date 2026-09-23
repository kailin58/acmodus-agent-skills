
# 化妆模块POC:FaceMesh嘴唇mask→口红上色(区域感知色彩混合)
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import pyarrow.parquet as pq
from facemesh_lib.tasks import python as mp_python
from facemesh_lib.tasks.python import vision as mp_vision
from facemesh_lib import Image as MPImage, ImageFormat

t = pq.ParquetFile(r"B:\mode\vton_training\data\VITON-HD-Captions\data\train-00000-of-00012.parquet").read_row_group(0)
b = t.column('image')[0].as_py()['bytes']
img = Image.open(io.BytesIO(b)).convert("RGB")
img.thumbnail((440,440))
W, H = img.size
arr = np.asarray(img).astype(np.float32)

det = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=r"B:\mode\models\face_landmarker.task"), num_faces=1))
r = det.detect(MPImage(image_format=ImageFormat.SRGB, data=arr.astype(np.uint8)))
lm = np.array([[p.x*W, p.y*H] for p in r.face_landmarks[0]])

# 嘴唇外圈(FaceMesh标准索引:外唇0-16环+内唇78-95)
outer = [61,185,40,39,37,0,267,269,270,409,291,308,415,310,311,312,13,82,81,80,191]
inner = [78,191,80,81,82,13,312,311,310,415,308,291,409,270,269,267,0,37,39,40,185]
pts = np.array([lm[i] for i in outer], np.int32)

# 嘴唇mask(外圈多边形,羽化)
mask = Image.new("L", (W,H), 0)
ImageDraw.Draw(mask).polygon(pts.reshape(-1).tolist(), fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(3))
m = np.asarray(mask).astype(np.float32)/255.0

# 口红色(正红#C2185B)soft-light/混合
lip_rgb = np.array([194, 24, 91], np.float32)
# 保留原唇纹理:色相替换+保留明度
gray = arr.mean(2, keepdims=True)/255.0
tint = lip_rgb[None,None,:]/255.0 * (0.35 + gray*1.3)  # 明度调制
out = arr*(1-m[...,None]*0.75) + tint*255*m[...,None]*0.75
out = np.clip(out, 0, 255).astype(np.uint8)

res = Image.fromarray(out)
try: font=ImageFont.truetype("arial.ttf",18)
except: font=ImageFont.load_default()
ImageDraw.Draw(res).text((15,15), "MAKEUP POC: lipstick #C2185B", fill=(255,255,255), font=font)
res.save(r"B:\mode\makeup_poc_lipstick.png")
print("MAKEUP_POC_DONE", flush=True)
