
# F6:用户脸→Anny头部形变→虚拟自己渲染
import os, math
import numpy as np
import torch
from PIL import Image, ImageDraw
import body_engine as anny

lm = np.load(r"B:\mode\vton_training\face_lm.npy")  # (478,3) 归一化坐标

# ①脸型参数(与基准脸的偏差)
def face_params(lm):
    fw = abs(lm[356][0]-lm[127][0])          # 脸宽(耳前)
    fh = abs(lm[152][1]-lm[10][1])           # 脸长(下巴-额头)
    jaw_w = abs(lm[172][0]-lm[397][0])       # 下巴宽
    cheek_w = abs(lm[234][0]-lm[454][0])     # 颧骨宽
    nose_w = abs(lm[129][0]-lm[358][0])
    eye_d = abs(lm[133][0]-lm[362][0])
    chin_len = abs(lm[152][1]-lm[168][1])    # 下巴长度(鼻梁到下巴)
    return {
        "aspect": fh/max(fw,1e-6),            # 脸长宽比(瘦长/圆润)
        "jaw": jaw_w/max(cheek_w,1e-6),        # 下巴/颧骨(尖脸/方脸)
        "nose": nose_w/max(fw,1e-6),
        "eye": eye_d/max(fw,1e-6),
        "chin": chin_len/max(fh,1e-6),
    }
P = face_params(lm)
print("脸型参数:", {k: round(v,3) for k,v in P.items()}, flush=True)

# ②Anny人体+头部形变
model = body_engine.Engine(local_changes="default")
res0 = model(None, {"height": 0.5, "weight": 0.5})
V = np.asarray(res0["vertices"])[0].copy()
z_min, z_max = V[:,2].min(), V[:,2].max()
h = z_max - z_min
head_z = z_max - h*0.12   # 头部区域(顶部12%)
head_mask = V[:,2] > head_z
print("头部顶点:", head_mask.sum(), flush=True)

# 基准参数(平均脸)与形变函数
BASE = {"aspect": 1.35, "jaw": 0.62, "nose": 0.28, "eye": 0.22, "chin": 0.42}
scale = 0.35  # 形变幅度系数(不过度夸张)
for idx in np.where(head_mask)[0]:
    x, y, z = V[idx]
    t = (z - head_z) / (z_max - head_z + 1e-9)  # 头部内权重(头顶=1)
    t = t ** 1.2
    # 脸长宽比:瘦长→X收窄/Z拉伸;圆润→反向
    da = (P["aspect"] - BASE["aspect"]) * scale
    V[idx,0] *= (1 - da*0.5*t)
    # 下巴:尖→下半部X再收;方→放宽
    dj = (P["jaw"] - BASE["jaw"]) * scale
    low = max(0, (head_z + (z_max-head_z)*0.5 - z)) / (z_max-head_z+1e-9)  # 越靠下越1
    V[idx,0] *= (1 + dj*0.8*t*low)

# ③渲染对比(原始vs形变)
def render(Va, path, title):
    W, H = 360, 420
    img = Image.new("RGB", (W,H), (24,24,28))
    d = ImageDraw.Draw(img)
    z0, z1 = Va[:,2].min(), Va[:,2].max()
    x0, x1 = Va[:,0].min(), Va[:,0].max()
    # 只渲染头部+肩(上部40%)
    zz = Va[:,2]
    sel = zz > (z1 - h*0.35)
    Vs = Va[sel]
    vs0, vs1 = Vs[:,2].min(), Vs[:,2].max()
    vx0, vx1 = Vs[:,0].min(), Vs[:,0].max()
    px = ((Vs[:,0]-vx0)/(vx1-vx0+1e-9)*(W-80)+40).astype(int)
    py = ((vs1-Vs[:,2])/(vs1-vs0+1e-9)*(H-100)+60).astype(int)
    for i in range(len(Vs)):
        d.ellipse((px[i]-1,py[i]-1,px[i]+1,py[i]+1), fill=(150,190,255))
    d.text((20,20), title, fill=(255,255,255))
    img.save(path)

render(np.asarray(res0["vertices"])[0], r"B:\mode\vton_training\garment_poc\avatar_before.png", "BEFORE(基准脸)")
render(V, r"B:\mode\vton_training\garment_poc\avatar_after.png", f"AFTER(用户脸型 aspect={P['aspect']:.2f} jaw={P['jaw']:.2f})")
print("F6_RENDER_DONE", flush=True)
