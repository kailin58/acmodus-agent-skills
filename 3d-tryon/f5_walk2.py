
# F5:3D人体行走动画+红裙纹理跟随→GIF
import os, math
import numpy as np
import torch
from PIL import Image, ImageDraw
import anny

OUT = r"B:\mode\vton_training\garment_poc"
FRAMES_DIR = os.path.join(OUT, "walk_frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

model = anny.Anny(local_changes="default")
NB = model.bone_count
print("骨骼数:", NB, flush=True)

# 静态基准:res["rest_bone_poses"]或全identity下顶点
pose0 = torch.eye(4, dtype=torch.float64)[None, None].repeat(1, NB, 1, 1)
res0 = model(pose0, {"height": 0.5, "weight": 0.5})
V0 = np.asarray(res0["vertices"])[0]  # 静止顶点
bone_rest = np.asarray(res0["rest_bone_poses"])[0]  # (104,4,4)静息骨骼变换
bh = np.asarray(res0["rest_bone_heads"])[0]  # (104,3)骨骼头位置
print("bone_heads:", bh.shape, "| Z范围", bh[:,2].min().round(2), bh[:,2].max().round(2), flush=True)

# 找腿骨/臂骨:Z低(腿)=z< -0.3;X偏(臂)|x|>0.3
leg_bones = [i for i in range(NB) if bh[i,2] < -0.25 and abs(bh[i,0]) < 0.25]
arm_bones = [i for i in range(NB) if abs(bh[i,0]) > 0.30]
print("腿骨:", len(leg_bones), "臂骨:", len(arm_bones), flush=True)

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def frame_pose(t):
    """行走循环:双腿反相摆动+双臂反相摆动+轻微身体起伏"""
    pose = torch.eye(4, dtype=torch.float64)[None, None].repeat(1, NB, 1, 1)
    P = pose[0].numpy()
    swing = math.sin(t) * 0.45
    for b in leg_bones:
        side = 1 if bh[b,0] > 0 else -1
        ang = swing * side if abs(bh[b,2]) < 0.1 else swing * side * 0.5  # 膝踝减幅
        # 局部旋转(相对父骨骼),让anny FK链处理
        P[b][:3,:3] = rot_x(ang) @ P[b][:3,:3]
    for b in arm_bones:
        side = 1 if bh[b,0] > 0 else -1
        ang = -swing * side * 0.6
        P[b][:3,:3] = rot_x(ang) @ P[b][:3,:3]
    return torch.from_numpy(P)[None]

# 红裙纹理映射参数(和f3_v5一致)
z_min, z_max = V0[:,2].min(), V0[:,2].max()
x_min, x_max = V0[:,0].min(), V0[:,0].max()
h = z_max - z_min
shoulder_z = z_max - h*0.15
hem_z = z_max - h*0.62
G = Image.open(os.path.join(OUT, "dress_red.png")).convert("RGB").resize((200, 260))
garr = np.array(G)
gw, gh = 200, 260

def colorize(V):
    colors = np.full((len(V), 3), 90, np.uint8)
    for idx in range(len(V)):
        z = V[idx,2]
        if not (hem_z < z < shoulder_z): continue
        layer = V[np.abs(V[:,2] - z) < h*0.012]
        if len(layer) < 3: continue
        lx_min, lx_max = layer[:,0].min(), layer[:,0].max()
        x = V[idx,0]
        u = int(np.clip((x-lx_min)/(lx_max-lx_min+1e-9)*(gw-1),0,gw-1))
        v = int(np.clip((shoulder_z-z)/(shoulder_z-hem_z)*(gh-1),0,gh-1))
        r_,g_,b_ = garr[v,u]
        if r_>225 and g_>225 and b_>225: continue
        colors[idx] = (r_,g_,b_)
    return colors

# 12帧行走循环
N_FRAMES = 12
frames = []
for fi in range(N_FRAMES):
    t = fi / N_FRAMES * 2 * math.pi
    pose = frame_pose(t)
    res = model(pose, {"height": 0.5, "weight": 0.5})
    V = np.asarray(res["vertices"])[0]
    colors = colorize(V)
    # 渲染(正交正视+轻微侧转15°显摆臂)
    ang = math.radians(12)
    Vr = V.copy()
    xr = Vr[:,0]*math.cos(ang) - Vr[:,1]*math.sin(ang)
    yr = Vr[:,0]*math.sin(ang) + Vr[:,1]*math.cos(ang)
    W, H2 = 300, 560
    img = Image.new("RGB", (W, H2), (20,20,24))
    d = ImageDraw.Draw(img)
    rx_min, rx_max = xr.min(), xr.max()
    rz_min, rz_max = Vr[:,2].min(), Vr[:,2].max()
    order = np.argsort(yr)
    px = ((xr-rx_min)/(rx_max-rx_min)*(W-50)+25).astype(int)
    py = ((rz_max-Vr[:,2])/(rz_max-rz_min)*(H2-50)+25).astype(int)
    for i in order:
        d.ellipse((px[i]-1,py[i]-1,px[i]+1,py[i]+1), fill=tuple(int(c) for c in colors[i]))
    frames.append(img)
    img.save(os.path.join(FRAMES_DIR, f"frame_{fi:02d}.png"))
    print(f"FRAME {fi}", flush=True)

frames[0].save(os.path.join(OUT, "walk.gif"), save_all=True, append_images=frames[1:], duration=120, loop=0)
print("WALK_GIF_DONE", flush=True)
