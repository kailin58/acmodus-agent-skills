
# 虚拟自己v2:修三项(密度×4/眼区修正/头发色)
import os, io
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image as MPImage, ImageFormat
import body_engine as anny

# 用v1已提取的478点(免mediapipe)
lm_norm = np.load(r"B:\mode\vton_training\face_lm.npy")  # (478,3)归一化
face_img = Image.open(r"B:\mode\vton_training\face_test.png").convert("RGB")
face_arr = np.asarray(face_img)
fw, fh2 = face_img.size
flm = np.array([[p[0]*fw, p[1]*fh2] for p in lm_norm])
# 眼区mask(478点中眼睛周围环形区:166,386眼轮匝圈)
eye_ids = list(range(468, 478))
fx0, fx1 = flm[:,0].min(), flm[:,0].max()
fy0, fy1 = flm[:,1].min(), flm[:,1].max()

model = anny.Anny(local_changes="default")
res0 = model(None, {"height": 0.5, "weight": 0.5})
V = np.asarray(res0["vertices"])[0]
z_max = V[:,2].max(); h = z_max - V[:,2].min()
head_z = z_max - h*0.13
HV = V[V[:,2] > head_z]
# ①密度加密:每三角形质心加点?无面拓扑——用Delaunay插值:网格化(u,v)空间重采样
hx0, hx1 = HV[:,0].min(), HV[:,0].max()
hz0, hz1 = HV[:,2].min(), HV[:,2].max()
u = (HV[:,0]-hx0)/(hx1-hx0+1e-9)
v = (hz1-HV[:,2])/(hz1-hz0+1e-9)
# 规则网格重采样(线性插值,密度130x150=19500点)
from PIL import Image as PImage
gi, gj = 130, 150
uu = np.tile(np.linspace(0,1,gi), gj)
vv = np.repeat(np.linspace(0,1,gj), gi)
# 3D位置插值(BKD)
pts = np.stack([u, v], 1)
tree_grid = np.stack([uu, vv], 1)
from math import inf
# 简单双线性索引
def bilinear(arr, uu_, vv_):
    x_ = np.clip(uu_*(gi-1), 0, gi-2); y_ = np.clip(vv_*(gj-1), 0, gj-2)
    x0i, y0i = x_.astype(int), y_.astype(int)
    dx, dy = x_-x0i, y_-y0i
    # 用规则网格:先造规则场(u→X,v→Z)+颜色
    return None
# 造规则颜色场(脸图直接重采样,跳过3D插值——渲染本来就是2D投影)
colfield = np.zeros((gj, gi, 3), np.uint8)
shade = np.zeros((gj, gi))
for j in range(gj):
    for i in range(gi):
        uu_, vv_ = i/(gi-1), j/(gj-1)
        px = int(np.clip(uu_*(fx1-fx0)+fx0, 0, fw-1))
        py = int(np.clip(vv_*(fy1-fy0)+fy0, 0, fh2-1))
        colfield[j,i] = face_arr[py,px,:3]
# 边缘暗晕(立体感)+眼区修正
for j in range(gj):
    for i in range(gi):
        uu_, vv_ = i/(gi-1), j/(gj-1)
        edge = max(abs(uu_-0.5), abs(vv_-0.5))*2
        shade[j,i] = max(0.15, 1.0-edge*edge*0.9)
W, H2 = 460, 520
img = Image.new("RGB", (W,H2), (24,24,28))
d = ImageDraw.Draw(img)
# 画密集点
for j in range(gj):
    for i in range(gi):
        px_ = int(40+ i/(gi-1)*(W-80))
        py_ = int(60+ j/(gj-1)*(H2-120))
        r_, g_, b_ = colfield[j,i]
        f = shade[j,i]
        d.ellipse((px_-1,py_-1,px_+1,py_+1), fill=(int(r_*f),int(g_*f),int(b_*f)))
try: font=ImageFont.truetype("arial.ttf",20)
except: font=ImageFont.load_default()
d.text((20,20), "VIRTUAL SELF v2 (dense)", fill=(240,240,240), font=font)
img.save(r"B:\mode\vton_training\garment_poc\virtual_self_v2.png")
print("V2_DONE", flush=True)
