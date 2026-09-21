
# F3-v4:高对比渲染(红裙+暗底+大点)
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
x_min, x_max = V[:,0].min(), V[:,0].max()
h = z_max - z_min
shoulder_z = z_max - h*0.16
hip_z = z_max - h*0.50

# 用SD生成的红裙图(对比度高)
import glob
cands = glob.glob(os.path.join(OUT, "dress_red.png"))
src = cands[0]
G = Image.open(src).convert("RGB").resize((200, 260))
garr = np.array(G)

colors = np.full((len(V), 3), 90, np.uint8)  # 人体=暗灰
mask_front = (V[:,2] < shoulder_z) & (V[:,2] > hip_z)
hit = 0
for idx in np.where(mask_front)[0]:
    x, z = V[idx, 0], V[idx, 2]
    u = int(np.clip((x - x_min)/(x_max - x_min)*(gw-1), 0, gw-1)) if (gw := 200) else 0
    v = int(np.clip((shoulder_z - z)/(shoulder_z - hip_z)*(gh-1), 0, gh-1)) if (gh := 260) else 0
    r, g, b = garr[v, u]
    if r > 235 and g > 235 and b > 235: continue
    colors[idx] = (r, g, b); hit += 1
print(f"着色: {hit}/{mask_front.sum()}", flush=True)

W, H = 360, 640
img = Image.new("RGB", (W, H), (20, 20, 24))
d = ImageDraw.Draw(img)
px = ((V[:,0]-x_min)/(x_max-x_min)*(W-60)+30).astype(int)
py = ((z_max-V[:,2])/(z_max-z_min)*(H-60)+30).astype(int)
order = np.argsort(V[:,1])
for i in order:
    c = tuple(int(x) for x in colors[i])
    d.ellipse((px[i]-1, py[i]-1, px[i]+1, py[i]+1), fill=c)
img.save(os.path.join(OUT, "fit_v4.png"))
print("V4_SAVED", flush=True)
