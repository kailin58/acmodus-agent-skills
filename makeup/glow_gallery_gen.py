
# Vyton-Glow v2:多妆容批量渲染(4场景×3色号=12组合,存素材库)
import io, os, json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import pyarrow.parquet as pq
from facemesh_lib.tasks import python as mp_python
from facemesh_lib.tasks.python import vision as mp_vision
from facemesh_lib import Image as MPImage, ImageFormat

OUT = r"B:\mode\glow_gallery"
os.makedirs(OUT, exist_ok=True)

# 妆容知识库v0的场景参数
SCENES = {
    "通勤": {"lip": (181,103,107), "eye": (196,168,141), "blush": 0.25, "brow": 0.35},
    "约会": {"lip": (194,24,91), "eye": (166,123,158), "blush": 0.4, "brow": 0.45},
    "舞台": {"lip": (142,14,60), "eye": (122,78,140), "blush": 0.55, "brow": 0.55},
    "日常": {"lip": (217,142,150), "eye": (217,194,184), "blush": 0.2, "brow": 0.3},
}
LIP_VARIANTS = {"正红": (194,24,91), "豆沙": (166,110,116), "珊瑚": (226,120,104)}

t = pq.ParquetFile(r"B:\mode\vton_training\data\VITON-HD-Captions\data\train-00000-of-00012.parquet").read_row_group(0)
b = t.column('image')[0].as_py()['bytes']
img0 = Image.open(io.BytesIO(b)).convert("RGB"); img0.thumbnail((440,440))
W, H = img0.size
arr0 = np.asarray(img0).astype(np.float32)

det = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=r"B:\mode\models\face_landmarker.task"), num_faces=1))
r0 = det.detect(MPImage(image_format=ImageFormat.SRGB, data=arr0.astype(np.uint8)))
lm = np.array([[p.x*W, p.y*H] for p in r0.face_landmarks[0]])

def region_mask(ids, blur):
    m = Image.new("L", (W,H), 0)
    pts = np.array([lm[i] for i in ids], np.int32)
    ImageDraw.Draw(m).polygon(pts.reshape(-1).tolist(), fill=255)
    return np.asarray(m.filter(ImageFilter.GaussianBlur(blur))).astype(np.float32)/255.0

LIP = [61,185,40,39,37,0,267,269,270,409,291,308,415,310,311,312,13,82,81,80,191]
EYE = [70,63,105,66,107,55,193,336,296,334,293,300]
CHEEK_L = [234,93,132,58,172,136,150,149,176,148]
CHEEK_R = [454,323,361,288,397,365,379,378,400,377]
BROW = [70,63,105,66,107,336,296,334,293]

m_eye = region_mask(EYE, 6)
m_bl = region_mask(CHEEK_L, 12); m_br = region_mask(CHEEK_R, 12)
m_brow = region_mask(BROW, 3)

def apply(arr, scene, lip_rgb):
    out = arr.copy()
    def blend(mask, rgb, s):
        gray = out.mean(2, keepdims=True)/255.0
        tint = np.array(rgb, np.float32)[None,None,:]/255.0*(0.4+gray*1.2)
        out[:] = out*(1-mask[...,None]*s) + tint*255*mask[...,None]*s
    blend(region_mask(LIP, 3), lip_rgb, 0.75)
    blend(m_eye, scene["eye"], 0.45)
    blend(m_bl, (235,130,140), scene["blush"]); blend(m_br, (235,130,140), scene["blush"])
    blend(m_brow, (70,45,30), scene["brow"])
    return np.clip(out,0,255).astype(np.uint8)

count = 0
for sname, sc in SCENES.items():
    for vname, lip in LIP_VARIANTS.items():
        res = Image.fromarray(apply(arr0, sc, lip))
        res.save(os.path.join(OUT, f"glow_{sname}_{vname}.png"))
        count += 1
print("GLOW_GALLERY:", count, flush=True)
