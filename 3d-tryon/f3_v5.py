
# F3-v5:修髋部空洞+溢出(裙摆延伸+躯干X归一)
import os
import numpy as np
import torch
from PIL import Image, ImageDraw
import body_engine

OUT = r"B:\mode\vton_training\garment_poc"
model = body_engine.BodyEngine(local_changes="default")
pose = torch.eye(4, dtype=torch.float64)[None, None].repeat(1, model.bone_count, 1, 1)
res = model(pose, {"height": 0.5, "weight": 0.5})
V = np.asarray(res["vertices"])[0]
z_min, z_max = V[:,2].min(), V[:,2].max()
h = z_max - z_min
shoulder_z = z_max - h*0.15
hem_z = z_max - h*0.62        # 裙摆到大腿上段(修空洞)

# 躯干X宽度:逐Z切片采样(修溢出:不用全肩宽,用每层实际躯干宽)
G = Image.open(os.path.join(OUT, "dress_red.png")).convert("RGB").resize((200, 260))
garr = np.array(G)
gw, gh = 200, 260

colors = np.full((len(V), 3), 90, np.uint8)
hit = 0
in_zone = (V[:,2] < shoulder_z) & (V[:,2] > hem_z)
# 逐层躯干宽
zs = V[:,2]
for idx in np.where(in_zone)[0]:
    z = zs[idx]
    # 该Z层的躯干X范围(限制在人体上)
    layer = V[(np.abs(zs - z) < h*0.01)]
    lx_min, lx_max = layer[:,0].min(), layer[:,0].max()
    x = V[idx,0]
    u = int(np.clip((x - lx_min)/(lx_max - lx_min + 1e-9)*(gw-1), 0, gw-1))
    v = int(np.clip((shoulder_z - z)/(shoulder_z - hem_z)*(gh-1), 0, gh-1))
    r, g, b = garr[v, u]
    if r>225 and g>225 and b>225: continue
    colors[idx] = (r,g,b); hit += 1
print(f"着色: {hit}/{in_zone.sum()}", flush=True)

W, H = 360, 640
img = Image.new("RGB", (W, H), (20,20,24))
d = ImageDraw.Draw(img)
x_min, x_max = V[:,0].min(), V[:,0].max()
px = ((V[:,0]-x_min)/(x_max-x_min)*(W-60)+30).astype(int)
py = ((z_max-V[:,2])/(z_max-z_min)*(H-60)+30).astype(int)
for i in np.argsort(V[:,1]):
    d.ellipse((px[i]-1,py[i]-1,px[i]+1,py[i]+1), fill=tuple(int(c) for c in colors[i]))
img.save(os.path.join(OUT, "fit_v5.png"))
print("V5_SAVED", flush=True)
