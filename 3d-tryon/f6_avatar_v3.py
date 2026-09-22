
# F6 v3:法线着色+夸张对比+差异热图
import os, math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import body_engine as anny

lm = np.load(r"B:\mode\vton_training\face_lm.npy")
fw = abs(lm[356][0]-lm[127][0]); fh = abs(lm[152][1]-lm[10][1])
jaw_w = abs(lm[172][0]-lm[397][0]); cheek_w = abs(lm[234][0]-lm[454][0])
P = {"aspect": fh/max(fw,1e-6), "jaw": jaw_w/max(cheek_w,1e-6)}

model = body_engine.Engine(local_changes="default")
res0 = model(None, {"height": 0.5, "weight": 0.5})
V0 = np.asarray(res0["vertices"])[0]
z_max = V0[:,2].max(); h = z_max - V0[:,2].min()
head_z = z_max - h*0.12
head_mask = V0[:,2] > head_z

def deform(Va, scale):
    V = Va.copy()
    BASE = {"aspect": 1.35, "jaw": 0.62}
    for idx in np.where(V[:,2] > head_z)[0]:
        z = V[idx,2]
        t = ((z-head_z)/(z_max-head_z+1e-9))**1.2
        da = (P["aspect"]-BASE["aspect"])*scale
        V[idx,0] *= (1-da*0.5*t)
        dj = (P["jaw"]-BASE["jaw"])*scale
        low = max(0,(head_z+(z_max-head_z)*0.5-z))/(z_max-head_z+1e-9)
        V[idx,0] *= (1+dj*0.8*t*low)
    return V

V_soft = deform(V0, 0.35)   # 真实幅度
V_hard = deform(V0, 1.2)    # 夸张3.4倍(可视化用)

def normals_color(Vs, sel):
    """法线着色:顶点法线朝Z的分量→明暗"""
    P3 = Vs[sel]
    # 近邻法线(用网格结构近似:每个点的局部平面)——点云无面,用k近邻
    from scipy.spatial import cKDTree
    tree = cKDTree(P3)
    _, idxs = tree.query(P3, k=7)
    nz = np.zeros(len(P3))
    for i, nb in enumerate(idxs):
        pts = P3[nb]
        c = pts.mean(0)
        u, s, vt = np.linalg.svd(pts - c)
        n = vt[2]
        nz[i] = abs(n[2])
    return nz

def render(Vs, path, label, boost=1.0):
    W, H = 460, 520
    img = Image.new("RGB", (W,H), (24,24,28))
    d = ImageDraw.Draw(img)
    zz = Vs[:,2]
    sel = zz > (zz.max()-h*0.38)
    P3 = Vs[sel]
    nz = normals_color(Vs, sel)
    vs0, vs1 = P3[:,2].min(), P3[:,2].max()
    vx0, vx1 = P3[:,0].min(), P3[:,0].max()
    px = ((P3[:,0]-vx0)/(vx1-vx0+1e-9)*(W-90)+45).astype(int)
    py = ((vs1-P3[:,2])/(vs1-vs0+1e-9)*(H-110)+70).astype(int)
    shade = nz**0.6  # 提亮中间调
    for i in range(len(P3)):
        b = int(70+shade[i]*185*boost)
        d.ellipse((px[i]-1,py[i]-1,px[i]+1,py[i]+1), fill=(int(b*0.72),int(b*0.86),b))
    try: font = ImageFont.truetype("arial.ttf", 20)
    except: font = ImageFont.load_default()
    d.text((20,20), label, fill=(240,240,240), font=font)
    img.save(path)

OUT = r"B:\mode\vton_training\garment_poc"
render(V0, os.path.join(OUT,"av3_base.png"), "BASE(normal shade)")
render(V_soft, os.path.join(OUT,"av3_user.png"), "USER normal")
render(V_hard, os.path.join(OUT,"av3_user_boost.png"), "USER x3.4 (visual)")
# 差异热图:头部X方向偏移
W, H = 460, 520
img = Image.new("RGB", (W,H), (24,24,28))
d = ImageDraw.Draw(img)
sel = V0[:,2] > (V0[:,2].max()-h*0.38)
A, B = V0[sel], V_soft[sel]
diff = np.abs(B[:,0]-A[:,0])
dn = (diff-diff.min())/(diff.max()-diff.min()+1e-9)
vs0, vs1 = A[:,2].min(), A[:,2].max()
vx0, vx1 = A[:,0].min(), A[:,0].max()
px = ((A[:,0]-vx0)/(vx1-vx0+1e-9)*(W-90)+45).astype(int)
py = ((vs1-A[:,2])/(vs1-vs0+1e-9)*(H-110)+70).astype(int)
for i in range(len(A)):
    v = dn[i]
    col = (int(40+v*215), int(40+v*60), int(90-v*60) if v<0.4 else 40)
    d.ellipse((px[i]-1,py[i]-1,px[i]+1,py[i]+1), fill=col)
try: font = ImageFont.truetype("arial.ttf", 20)
except: font = ImageFont.load_default()
d.text((20,20), "DEFORMATION HEATMAP", fill=(240,240,240), font=font)
img.save(os.path.join(OUT,"av3_heatmap.png"))
print("F6_V3_DONE", flush=True)
