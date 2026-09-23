
# 化妆扩展:眼影+腮红+眉毛(同管线换区域)
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

def region_mask(poly_ids, blur=4):
    m = Image.new("L", (W,H), 0)
    pts = np.array([lm[i] for i in poly_ids], np.int32)
    ImageDraw.Draw(m).polygon(pts.reshape(-1).tolist(), fill=255)
    return np.asarray(m.filter(ImageFilter.GaussianBlur(blur))).astype(np.float32)/255.0

def blend(arr, mask, rgb, strength=0.55, mode="tone"):
    gray = arr.mean(2, keepdims=True)/255.0
    tint = np.array(rgb, np.float32)[None,None,:]/255.0*(0.4+gray*1.2)
    return arr*(1-mask[...,None]*strength) + tint*255*mask[...,None]*strength

out = arr.copy()
# 眼影(上眼睑:眼眶上缘弧)
eyeshadow_ids = [70,63,105,66,107,55,193,168,417,285,282,336,296,334,293,300]  # 眉下至眼褶
out = blend(out, region_mask(eyeshadow_ids, 6), [130, 88, 160], 0.5)   # 紫棕眼影
# 腮红(颧骨:椭圆围绕耳前到脸颊)
cheek_ids = [234,93,132,58,172,136,150,149,176,148]  # 左颧弧
out = blend(out, region_mask(cheek_ids, 12), [235, 130, 140], 0.35)   # 粉腮红(淡)
cheek_ids2 = [454,323,361,288,397,365,379,378,400,377]  # 右颧弧
out = blend(out, region_mask(cheek_ids2, 12), [235, 130, 140], 0.35)
# 眉毛加深
brow_ids = [70,63,105,66,71]  
bm = region_mask([70,63,105,66,107,336,296,334,293], 3)
out = blend(out, bm, [70, 45, 30], 0.45)  # 深棕眉

res = Image.fromarray(np.clip(out,0,255).astype(np.uint8))
try: font=ImageFont.truetype("arial.ttf",18)
except: font=ImageFont.load_default()
ImageDraw.Draw(res).text((15,15), "MAKEUP v2: eyeshadow+blush+brow", fill=(255,255,255), font=font)
res.save(r"B:\mode\makeup_poc_v2_full.png")
print("MAKEUP_V2_DONE", flush=True)
