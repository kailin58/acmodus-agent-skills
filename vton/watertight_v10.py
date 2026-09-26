# v10:面积加权法线重算+统一朝外+Taubin防收缩+量化验收
import numpy as np
from PIL import Image, ImageDraw
import math
from collections import defaultdict

V, F = [], []
for l in open(r"B:\mode\vton_training\mesh_50\sample_0000_watertight_v8.obj"):
    if l.startswith("v "): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith("f "): F.append([int(x)-1 for x in l.split()[1:4]])
V = np.array(V); F = np.array(F)

adj = defaultdict(set)
for f in F:
    for a in range(3):
        adj[f[a]].add(f[(a+1)%3]); adj[f[a]].add(f[(a+2)%3])

# Taubin 15轮(λ=0.5, μ=-0.53,防收缩)
for it in range(15):
    lam = 0.5 if it % 2 == 0 else -0.53
    Vn = V.copy()
    for i, nbrs in adj.items():
        if len(nbrs):
            Vn[i] = V[i] + lam*(V[list(nbrs)].mean(0) - V[i])
    V = Vn

# 面积加权法线+统一朝外(物体中心取向)
def face_normal(a, b, c):
    n = np.cross(V[b]-V[a], V[c]-V[a])
    ln = np.linalg.norm(n)
    return n/ln if ln > 1e-12 else n*0, ln

FN = np.zeros((len(F), 3)); FA = np.zeros(len(F))
for fi, (a,b,c) in enumerate(F):
    n, area = face_normal(a,b,c)
    FN[fi] = n; FA[fi] = area
center = V[F].mean(1).mean(0)
for fi in range(len(F)):
    fc = V[F[fi]].mean(0)
    if FN[fi] @ (fc - center) < 0:
        FN[fi] = -FN[fi]
        F[fi] = [F[fi][0], F[fi][2], F[fi][1]]  # 翻转绕向
# 顶点法线(面积加权)
VN = np.zeros_like(V)
for fi in range(len(F)):
    for idx in F[fi]:
        VN[idx] += FN[fi]*FA[fi]
VN = VN/(np.linalg.norm(VN,axis=1,keepdims=True)+1e-9)

with open(r"B:\mode\vton_training\mesh_50\sample_0000_watertight_v10.obj", "w") as fo:
    for v in V: fo.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
    for a,b,c in F: fo.write(f"f {a+1} {b+1} {c+1}\n")

# 量化验收:相邻面法线偏差(度)分布——条纹=环缝处大偏差
devs = []
for fi in range(len(F)):
    for k in range(3):
        fj = None
        # 简化:邻面=共享顶点的面
# 更快:顶点法线与面法线偏差统计
vd = np.degrees(np.arccos(np.clip((VN[F].mean(1)/(np.linalg.norm(VN[F].mean(1),axis=1,keepdims=True)+1e-9)) @ np.array([0,0,1.0])*0 + np.einsum("ij,ij->i", VN[F].mean(1)/(np.linalg.norm(VN[F].mean(1),axis=1,keepdims=True)+1e-9), FN), -1, 1)))
print("法线一致性: mean", round(vd.mean(),1), "° | p95", round(np.percentile(vd,95),1), "° | >30°占比", round((vd>30).mean()*100,1), "%")

LIGHT = np.array([0.45,0.55,0.70]); LIGHT/=np.linalg.norm(LIGHT)
def render(view, light_dir, W2=320, H2=380, scale=3, title=""):
    L = np.array(light_dir); L/=np.linalg.norm(L)
    img = Image.new("RGB", (W2*scale,H2*scale),(16,16,22)); dr = ImageDraw.Draw(img)
    ctr = V[F].mean(1)
    if view=="front": Vs=V; VNs=VN; key=ctr[:,2]; ax1=0
    elif view=="side": Vs=V; VNs=VN; key=-np.abs(ctr[:,0]); ax1=2
    else:
        ang=math.radians(35); rot=np.array([[math.cos(ang),0,-math.sin(ang)],[0,1,0],[math.sin(ang),0,math.cos(ang)]])
        Vs=V@rot.T; VNs=VNs2=VNs if False else VN@rot.T; key=(Vs[F].mean(1))[:,2]; ax1=0
    order=np.argsort(key)
    fv=VNs[F].mean(1); fv=fv/(np.linalg.norm(fv,axis=1,keepdims=True)+1e-9)
    diffuse=np.clip(fv@L,0,1)
    facing=fv@np.array([0,0,1.0])
    rim=np.clip(1-np.abs(facing),0,1)**2*0.4
    spec=np.clip(fv@L,0,1)**32*0.25
    for fi in order:
        f3=F[fi]
        pts=[(int((Vs[i][ax1]+0.45)*(W2-40)*scale), int((0.52-Vs[i][1])*(H2-50)*scale)) for i in f3]
        b=0.15+0.80*diffuse[fi]+rim[fi]+spec[fi]
        c=int(255*min(b,1.0)**0.95)
        dr.polygon(pts, fill=(int(c*0.82),int(c*0.92),int(c*0.88)))
    out = img.resize((W2,H2), Image.LANCZOS)
    ImageDraw.Draw(out).text((8,4), title, fill=(255,255,70))
    return out
canvas = Image.new("RGB", (1000,410),(14,14,20))
canvas.paste(render("side", [0.9,0.1,0.4], title="GRAZING"), (12,10))
canvas.paste(render("3q", [0.45,0.55,0.70], title="soft 3Q"), (342,10))
canvas.paste(render("front", [0.0,0.3,0.95], title="front"), (672,10))
canvas.save(r"B:\mode\vton_training\mesh_50\watertight_v10_preview.png")
print("V10_DONE")
