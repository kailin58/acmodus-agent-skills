
# F6最后一环:用户脸纹理投影到3D头部(虚拟自己"像自己")
import os, io, math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image as MPImage, ImageFormat
import body_engine as anny

# ①测试脸照片
t = pq.ParquetFile(r"B:\mode\vton_training\data\VITON-HD-Captions\data\train-00000-of-00012.parquet").read_row_group(0)
b = t.column('image')[0].as_py()['bytes']
face_img = Image.open(io.BytesIO(b)).convert("RGB")
face_img.thumbnail((440,440))
face_arr = np.asarray(face_img)

# ②脸部关键点(478)
fdet = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=r"B:\mode\models\face_landmarker.task"), num_faces=1))
fr = fdet.detect(MPImage(image_format=ImageFormat.SRGB, data=face_arr))
if not fr.face_landmarks:
    print("NO_FACE"); exit()
flm = np.array([[p.x*face_img.width, p.y*face_img.height] for p in fr.face_landmarks[0]])  # 像素坐标

# ③3D人体+头部
model = body_engine.Engine(local_changes="default")
res0 = model(None, {"height": 0.5, "weight": 0.5})
V = np.asarray(res0["vertices"])[0]
z_max = V[:,2].max(); h = z_max - V[:,2].min()
head_z = z_max - h*0.13
head_mask = V[:,2] > head_z
HV = V[head_mask]
print("头部顶点:", len(HV), flush=True)

# ④投影映射:3D头部顶点(X,Z)→脸图(u,v)
# 3D头:X左右/Z上下;脸图:u左右/v上下
hx0, hx1 = HV[:,0].min(), HV[:,0].max()
hz0, hz1 = HV[:,2].min(), HV[:,2].max()
# 脸bbox(478点)
fx0, fx1 = flm[:,0].min(), flm[:,0].max()
fy0, fy1 = flm[:,1].min(), flm[:,1].max()

colors = np.zeros((len(HV), 3), np.uint8)
farr = np.asarray(face_img)
fw, fh2 = face_img.size
hit = 0
for i, v in enumerate(HV):
    # 归一(X→u, Z→v)
    u = (v[0]-hx0)/(hx1-hx0+1e-9)
    vv = (hz1-v[2])/(hz1-hz0+1e-9)
    # 只投影正面(局部Y朝前的区域用近似:先全投,v1验证)
    px = int(np.clip(u*(fx1-fx0)+fx0, 0, fw-1))
    py = int(np.clip(vv*(fy1-fy0)+fy0, 0, fh2-1))
    colors[i] = farr[py, px, :3]
    hit += 1
print(f"投影: {hit}/{len(HV)}", flush=True)

# ⑤渲染带脸纹理的头部(法线着色×脸纹理)
nz = np.zeros(len(HV))
y_mid = (HV[:,1].max()+HV[:,1].min())/2
for i in range(len(HV)):
    # 简化着色:面向前方(局部Y小=前)亮
    nz[i] = 1.0 - abs(HV[i,1]-y_mid)/(abs(HV[:,1]-y_mid).max()+1e-9)
nz = nz**0.7

W, H2 = 460, 520
img = Image.new("RGB", (W,H2), (24,24,28))
d = ImageDraw.Draw(img)
vs0, vs1 = HV[:,2].min(), HV[:,2].max()
vx0, vx1 = HV[:,0].min(), HV[:,0].max()
px = ((HV[:,0]-vx0)/(vx1-vx0+1e-9)*(W-90)+45).astype(int)
py = ((vs1-HV[:,2])/(vs1-vs0+1e-9)*(H2-110)+70).astype(int)
shade = nz**0.5
for i in range(len(HV)):
    r, g, bb = colors[i]
    f = 0.35 + shade[i]*0.65  # 纹理×法线
    d.ellipse((px[i]-1,py[i]-1,px[i]+1,py[i]+1),
              fill=(int(r*f), int(g*f), int(bb*f)))
try: font=ImageFont.truetype("arial.ttf",20)
except: font=ImageFont.load_default()
d.text((20,20), "VIRTUAL SELF v1 (face texture)", fill=(240,240,240), font=font)
img.save(r"B:\mode\vton_training\garment_poc\virtual_self_v1.png")
print("VIRTUAL_SELF_DONE", flush=True)
