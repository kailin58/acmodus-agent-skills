# 动作库扩容:劳动/展示动作(基础姿势+程序化动画组合)
import numpy as np, os
def read_obj(p):
    vs, fs = [], []
    for line in open(p, errors="replace"):
        if line.startswith("v "):
            _, a, b, c = line.split(); vs.append([float(a), float(b), float(c)])
        elif line.startswith("f "):
            fs.append(line.strip())
    return np.array(vs, np.float32), fs
v0, fs = read_obj(r"B:\mode\vton_training\pose_test\01_tpose.obj")
z = v0[:, 2]; H = z.max() - z.min(); hn = (z - z.min()) / H
def write(path, v):
    with open(path, "w") as f:
        for p in v: f.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
        for l in fs: f.write(l + "\n")
def bend(v, mask, dz, dy=0):
    out = v.copy()
    idx = np.where(mask)[0]
    if len(idx) == 0: return out
    out[idx, 2] += dz * np.linspace(0, 1, len(idx))
    out[idx, 1] += dy * np.linspace(0, 1, len(idx))
    return out
OUT = r"B:\mode\models\body3d\actions"
os.makedirs(OUT, exist_ok=True)
# 1.挥手:右臂(右上区)上下摆5帧
arm = (hn > 0.65) & (v0[:, 0] > np.median(v0[:, 0]) + 0.28 * (v0[:,0].max()-np.median(v0[:,0])))
for k, amp in enumerate([0.5, 1.0, 0.5, 0.0, 0.5]):
    v = v0.copy()
    v[arm, 0] += amp * 0.06
    v[arm, 1] += amp * 0.04
    write(rf"{OUT}\wave_{k}.obj", v)
# 2.鞠躬:上半身前倾6帧
upper = hn > 0.5
for k, ang in enumerate([0, 0.3, 0.6, 0.8, 0.6, 0.3]):
    v = v0.copy()
    v[upper, 1] -= ang * 0.15 * (hn[upper] - 0.5) * 2
    v[upper, 2] -= ang * 0.10 * (hn[upper] - 0.5) * 2
    write(rf"{OUT}\bow_{k}.obj", v)
# 3.下蹲:整体下压+膝弯近似4帧
for k, d in enumerate([0.0, 0.35, 0.6, 0.35]):
    v = v0.copy()
    leg = hn < 0.5
    v[leg, 2] -= d * 0.12
    v[:, 2] -= d * 0.06
    write(rf"{OUT}\squat_{k}.obj", v)
# 4.转身:整体绕Z轴旋转8帧(0-360)
for k in range(8):
    ang = k / 8 * 2 * np.pi
    v = v0.copy()
    cx, cy = np.median(v[:, 0]), np.median(v[:, 1])
    x, y = v[:, 0] - cx, v[:, 1] - cy
    v[:, 0] = cx + x * np.cos(ang) - y * np.sin(ang)
    v[:, 1] = cy + x * np.sin(ang) + y * np.cos(ang)
    write(rf"{OUT}\turn_{k}.obj", v)
print("ACTIONS_DONE:", len(os.listdir(OUT)))
