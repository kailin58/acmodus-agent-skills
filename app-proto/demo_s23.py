
import os, math, json
import numpy as np
import torch
from PIL import Image, ImageDraw
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

def colorize(V):
    colors = np.full((len(V),3), 90, np.uint8)
    for idx in range(len(V)):
        z = V[idx,2]
        if not (hem_z < z < shoulder_z): continue
        layer = V[np.abs(V[:,2]-z) < h*0.012]
        if len(layer)<3: continue
        lx0, lx1 = layer[:,0].min(), layer[:,0].max()
        u = int(np.clip((V[idx,0]-lx0)/(lx1-lx0+1e-9)*(gw-1),0,gw-1))
        v = int(np.clip((shoulder_z-z)/(shoulder_z-hem_z)*(gh-1),0,gh-1))
        r_,g_,b_ = garr[v,u]
        if r_>225 and g_>225 and b_>225: continue
        colors[idx] = (r_,g_,b_)
    return colors

def frame(V, colors, W=480, H2=720, big=True):
    fr = Image.new("RGB",(W,H2),(238,238,240))
    d = ImageDraw.Draw(fr)
    d.rectangle([0,int(H2*0.82),W,H2], fill=(212,212,216))
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

# ⑤静态试穿
frame(V0, colorize(V0)).save(os.path.join(OUT, "5_tryon.png"))
print("STEP2_TRYON_OK", flush=True)

# ⑥行走GIF
def rot_x(a):
    cc,ss = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,cc,-ss],[0,ss,cc]])
leg = [i for i in range(NB) if bh[i,2]<-0.25 and abs(bh[i,0])<0.25]
arm = [i for i in range(NB) if abs(bh[i,0])>0.30]
frames = []
for fi in range(6):
    tt = fi/6*2*math.pi
    P = np.repeat(np.eye(4,dtype=np.float64)[None], NB, axis=0)
    swing = math.sin(tt)*0.45
    for bn in leg:
        side = 1 if bh[bn,0]>0 else -1
        P[bn][:3,:3] = rot_x(swing*side*0.7) @ P[bn][:3,:3]
    for bn in arm:
        side = 1 if bh[bn,0]>0 else -1
        P[bn][:3,:3] = rot_x(-swing*side*0.6) @ P[bn][:3,:3]
    res = model(torch.from_numpy(P)[None], kw)
    V = np.asarray(res["vertices"])[0]
    frames.append(frame(V, colorize(V), 240, 360, big=False))
frames[0].save(os.path.join(OUT,"6_walk.gif"), save_all=True, append_images=frames[1:], duration=160, loop=0)
print("STEP3_WALK_OK", flush=True)
print("DEMO_ALL_DONE", flush=True)
