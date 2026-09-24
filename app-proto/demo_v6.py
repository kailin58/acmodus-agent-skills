
# Demo v6精装:胸口杂色多数滤波(红区内的白色/杂点→邻域多数色)
import os, math, json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import body_engine as anny
from scipy.spatial import cKDTree

OUT = r"B:\mode\app_proto\demo_output"
kw = json.load(open(os.path.join(OUT, "2_measure.json")))
model = anny.Anny(local_changes="default")
res0 = model(None, kw)
V0 = np.asarray(res0["vertices"])[0]
bh = np.asarray(res0["rest_bone_heads"])[0]
NB = model.bone_count
garr = np.array(Image.open(r"B:\mode\vton_training\garment_poc\dress_red.png").convert("RGB").resize((200,260)))
gw, gh = 200, 260
z_max = V0[:,2].max(); h = z_max - V0[:,2].min()
shoulder_z, hem_z = z_max-h*0.15, z_max-h*0.62
cx_all = np.median(V0[:,0])
arm_bones = [i for i in range(NB) if abs(bh[i,0])>0.30]
hand_pts = np.array([bh[i] for i in arm_bones if abs(bh[i,2]) < z_max - h*0.30])

def garment_core(V):
    idxs = []
    for idx in range(len(V)):
        z = V[idx,2]
        if not (hem_z < z < shoulder_z - h*0.03): continue
        dx = abs(V[idx,0]-cx_all)
        layer = V[np.abs(V[:,2]-z) < h*0.02]
        if len(layer)<5: continue
        dxs = np.sort(np.abs(layer[:,0]-cx_all))
        torso_half = dxs[int(len(dxs)*0.72)] + 1e-6
        if dx > torso_half: continue
        d_hand = np.min(np.linalg.norm(V[idx][None,:]-hand_pts, axis=1)) if len(hand_pts) else 9
        if d_hand < 0.28: continue
        idxs.append(idx)
    return np.array(idxs)

core = garment_core(V0)
# 多数滤波:core内每点颜色=其商品图采样色,但先检查采样色是否"离群白"→用邻域多数色替代
# 实现:先算全部core点的采样色,再KNN(k=9)取中值色替代
tree = cKDTree(V0[core])
colors_core = np.zeros((len(core),3), np.uint8)
for j, idx in enumerate(core):
    z = V0[idx,2]
    layer = V0[np.abs(V0[:,2]-z) < h*0.02]
    dxs = np.sort(np.abs(layer[:,0]-cx_all))
    torso_half = dxs[int(len(dxs)*0.72)] + 1e-6
    core_x = layer[np.abs(layer[:,0]-cx_all) <= torso_half][:,0]
    lx0, lx1 = core_x.min(), core_x.max()
    u = int(np.clip((V0[idx,0]-lx0)/(lx1-lx0+1e-9)*(gw-1),0,gw-1))
    v = int(np.clip((shoulder_z-z)/(shoulder_z-hem_z)*(gh-1),0,gh-1))
    colors_core[j] = garr[v,u,:3]
# KNN中值滤波(去胸口白斑)
d, nb_idx = tree.query(V0[core], k=9)
filtered = np.zeros_like(colors_core)
for j in range(len(core)):
    filtered[j] = np.median(colors_core[nb_idx[j]], axis=0).astype(np.uint8)
# 只替换"离群点"(与中值差大→白斑/杂色)
diff = np.abs(colors_core.astype(int) - filtered.astype(int)).sum(1)
replace_mask = diff > 90
colors_core[replace_mask] = filtered[replace_mask]
print("滤波替换:", int(replace_mask.sum()), "/", len(core), flush=True)

core_set = set(core.tolist())
color_map = {int(core[j]): tuple(int(c) for c in colors_core[j]) for j in range(len(core))}
def colorize(V, color_map):
    colors = np.full((len(V),3), 90, np.uint8)
    for idx, c in color_map.items():
        colors[idx] = c
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

frame(V0, colorize(V0, color_map)).save(os.path.join(OUT, "5_tryon_v6.png"))
print("V6_OK", flush=True)
