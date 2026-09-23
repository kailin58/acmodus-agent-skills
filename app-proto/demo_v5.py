
# Demo v5:腕部硬切(距手腕骨骼<0.28的红点全删)+肩部收紧
import os, math, json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import body_engine as anny

OUT = r"B:\mode\app_proto\demo_output"
kw = json.load(open(os.path.join(OUT, "2_measure.json")))
model = body_engine.Engine(local_changes="default")
res0 = model(None, kw)
V0 = np.asarray(res0["vertices"])[0]
bh = np.asarray(res0["rest_bone_heads"])[0]
NB = model.bone_count
garr = np.array(Image.open(r"B:\mode\vton_training\garment_poc\dress_red.png").convert("RGB").resize((200,260)))
gw, gh = 200, 260
z_max = V0[:,2].max(); h = z_max - V0[:,2].min()
shoulder_z, hem_z = z_max-h*0.15, z_max-h*0.62
cx_all = np.median(V0[:,0])

# 手腕/手部骨骼位置:手臂骨(|x|>0.30)中最远端的骨骼头(手)
arm_bones = [i for i in range(NB) if abs(bh[i,0])>0.30]
hand_pts = np.array([bh[i] for i in arm_bones if abs(bh[i,2]) < z_max - h*0.30])  # 手臂低位=腕/手
print("腕手骨骼数:", len(hand_pts), flush=True)

def garment_core(V):
    idxs = []
    for idx in range(len(V)):
        z = V[idx,2]
        if not (hem_z < z < shoulder_z - h*0.03): continue  # 肩部收紧3%
        dx = abs(V[idx,0]-cx_all)
        layer = V[np.abs(V[:,2]-z) < h*0.02]
        if len(layer)<5: continue
        dxs = np.sort(np.abs(layer[:,0]-cx_all))
        torso_half = dxs[int(len(dxs)*0.72)] + 1e-6  # 72分位再收紧
        if dx > torso_half: continue
        # 腕手硬切:距任一手部骨骼<0.28→删
        d_hand = np.min(np.linalg.norm(V[idx][None,:]-hand_pts, axis=1)) if len(hand_pts) else 9
        if d_hand < 0.28: continue
        idxs.append(idx)
    return np.array(idxs)

core = garment_core(V0)
core_set = set(core.tolist())
print("core:", len(core), flush=True)

def colorize(V, core_set):
    colors = np.full((len(V),3), 90, np.uint8)
    for idx in core_set:
        z = V[idx,2]
        layer = V[np.abs(V[:,2]-z) < h*0.02]
        dxs = np.sort(np.abs(layer[:,0]-cx_all))
        torso_half = dxs[int(len(dxs)*0.72)] + 1e-6
        core_x = layer[np.abs(layer[:,0]-cx_all) <= torso_half][:,0]
        lx0, lx1 = core_x.min(), core_x.max()
        u = int(np.clip((V[idx,0]-lx0)/(lx1-lx0+1e-9)*(gw-1),0,gw-1))
        v = int(np.clip((shoulder_z-z)/(shoulder_z-hem_z)*(gh-1),0,gh-1))
        colors[idx] = garr[v,u,:3]
    return colors

def frame(V, colors, W=480, H2=720, big=True):
    fr = Image.new("RGB",(W,H2),(238,238,240))
    d = ImageDraw.Draw(fr)
    d.rectangle([0,int(H2*0.82),W,H2], fill=(212,212,216))
    sh = Image.new("RGB",(W,H2),(0,0,0))
    ds = ImageDraw.Draw(sh)
    cx_img = W//2; fy = int(H2*0.85)
    ds.ellipse([cx_img-90,fy-12,cx_img+90,fy+24], fill=(118,118,122))
    fr = Image.blend(fr, sh.filter(ImageFilter.GaussianBlur(13)), 0.28)
    d = ImageDraw.Draw(fr)
    ang = math.radians(10)
    xr = V[:,0]*math.cos(ang)-V[:,1]*math.sin(ang)
    depth = V[:,0]*math.sin(ang)+V[:,1]*math.cos(ang)
    x0,x1 = xr.min(), xr.max(); z0,z1 = V[:,2].min(), V[:,2].max()
    px = ((xr-x0)/(x1-x0)*(W-160)+80).astype(int)
    py = ((z1-V[:,2])/(z1-z0)*(H2-140)+70).astype(int)
    for i in np.argsort(depth):
        if big:
            d.ellipse((px[i]-1,py[i]-1,px[i]+1,py[i]+1), fill=tuple(int(x) for x in colors[i]))
        else:
            d.point((px[i],py[i]), fill=tuple(int(x) for x in colors[i]))
    return fr

frame(V0, colorize(V0, core_set)).save(os.path.join(OUT, "5_tryon_v5.png"))
print("V5_TRYON_OK", flush=True)

def rot_x(a):
    cc,ss = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,cc,-ss],[0,ss,cc]])
leg = [i for i in range(NB) if bh[i,2]<-0.25 and abs(bh[i,0])<0.25]
frames = []
for fi in range(8):
    tt = fi/8*2*math.pi
    P = np.repeat(np.eye(4,dtype=np.float64)[None], NB, axis=0)
    swing = math.sin(tt)*0.45
    for bn in leg:
        side = 1 if bh[bn,0]>0 else -1
        P[bn][:3,:3] = rot_x(swing*side*0.7) @ P[bn][:3,:3]
    for bn in arm_bones:
        side = 1 if bh[bn,0]>0 else -1
        P[bn][:3,:3] = rot_x(-swing*side*0.6) @ P[bn][:3,:3]
    res = model(torch.from_numpy(P)[None], kw)
    V = np.asarray(res["vertices"])[0]
    frames.append(frame(V, colorize(V, core_set), 240, 360, big=False))
frames[0].save(os.path.join(OUT,"6_walk_v5.gif"), save_all=True, append_images=frames[1:], duration=150, loop=0)
print("V5_WALK_OK", flush=True)
