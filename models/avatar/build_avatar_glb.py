# avatar.glb v2: 22骨骼+蒙皮+walk动画(trimesh几何+pygltflib组装)
import json
import numpy as np
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Skin, Animation, AnimationSampler, AnimationChannel, AnimationChannelTarget

# 1.几何(复用之前解析)
vs, fs = [], []
import re
for line in open(r"B:\mode\vton_training\mesh_50\sample_0000.obj", errors="replace"):
    if line.startswith("v "):
        _, a, b, c = line.split(); vs.append([float(a), float(b), float(c)])
    elif line.startswith("f "):
        idx = []
        for p in line.split()[1:]:
            m = re.search(r"tensor\((\d+)\)", p)
            if m: idx.append(int(m.group(1)))
            else:
                try: idx.append(int(p.split("/")[0]))
                except ValueError: idx = []; break
        if idx and max(idx) <= len(vs):
            for k in range(1, len(idx)-1):
                fs.append([idx[0]-1, idx[k]-1, idx[k+1]-1])
V = np.array(vs, np.float32)
yup = V.copy(); yup[:, 1] = V[:, 2]; yup[:, 2] = -V[:, 1]  # Y-up
yup -= yup.mean(0); yup[:, 1] -= yup[:, 1].min()  # 居中+落地
F = np.array(fs, np.uint32)
print(f"几何: {len(V)}顶点 {len(F)}面")

# 2.骨骼(bones22_v2)+层级树
bones = json.load(open(r"B:\mode\models\body3d\bones22_v2.json"))
# 转到几何新坐标(居中偏移)
off = np.array(V).mean(0); off[1] = 0
for k, v in bones.items():
    if v:
        p = np.array(v, np.float32)
        p[1] = V[:, 2].mean() - 0  # 旧yup未居中,重算:旧yup的y=旧z
        bones[k] = None  # 重置,下面统一重定位
# 简化:直接用居中后的yup网格重定位骨骼
H = yup[:, 1].max() - yup[:, 1].min()
h = (yup[:, 1] - yup[:, 1].min()) / H
mid = np.median(yup[:, 0])
ALL = np.ones(len(yup), bool)
def band(hlo, hhi, xm):
    sel = (h > hlo) & (h < hhi) & xm
    return yup[sel].mean(0) if sel.any() else None
Lm = yup[:, 0] > mid + 0.03 * H
Rm = yup[:, 0] < mid - 0.03 * H
selLa = (h > 0.72) & (h < 0.80) & (yup[:, 0] > mid + 0.05 * H)
selRa = (h > 0.72) & (h < 0.80) & (yup[:, 0] < mid - 0.05 * H)
p33L, p66L = np.percentile(yup[selLa, 0], [33, 66])
p33R, p66R = np.percentile(yup[selRa, 0], [33, 66])
B = {
    "Hips": band(0.46, 0.52, ALL), "Spine": band(0.54, 0.58, ALL),
    "Spine1": band(0.60, 0.64, ALL), "Spine2": band(0.66, 0.71, ALL),
    "Neck": band(0.84, 0.88, ALL), "Head": band(0.92, 1.0, ALL),
    "LeftShoulder": band(0.74, 0.79, yup[:, 0] > mid + 0.02 * H),
    "RightShoulder": band(0.74, 0.79, yup[:, 0] < mid - 0.02 * H),
    "LeftArm": band(0.73, 0.79, (yup[:, 0] > mid + 0.05 * H) & (yup[:, 0] < p33L)),
    "RightArm": band(0.73, 0.79, (yup[:, 0] < mid - 0.05 * H) & (yup[:, 0] > p33R)),
    "LeftForeArm": band(0.73, 0.79, (yup[:, 0] >= p33L) & (yup[:, 0] < p66L)),
    "RightForeArm": band(0.73, 0.79, (yup[:, 0] >= p66R) & (yup[:, 0] < p33R)),
    "LeftHand": band(0.73, 0.79, yup[:, 0] >= p66L),
    "RightHand": band(0.73, 0.79, yup[:, 0] <= p66R),
    "LeftUpLeg": band(0.40, 0.48, Lm), "RightUpLeg": band(0.40, 0.48, Rm),
    "LeftLeg": band(0.20, 0.30, Lm), "RightLeg": band(0.20, 0.30, Rm),
    "LeftFoot": band(0.0, 0.08, Lm), "RightFoot": band(0.0, 0.08, Rm),
    "LeftToeBase": band(0.0, 0.04, Lm), "RightToeBase": band(0.0, 0.04, Rm),
    "HeadTop_End": band(0.97, 1.0, ALL),
}
names = [k for k, v in B.items() if v is not None]
missing = [k for k in B if B[k] is None]
if missing:
    print("定位None:", missing)
    # 兜底:右前臂区间手补(和v2一样)
    if "RightForeArm" in missing:
        sel = (h > 0.73) & (h < 0.79) & (yup[:, 0] >= p66R) & (yup[:, 0] <= p33R)
        B["RightForeArm"] = yup[sel].mean(0) if sel.any() else B["RightArm"]
        names.append("RightForeArm")
    for k in missing:
        if B[k] is None:
            B[k] = B["Hips"]  # 最终兜底
            if k not in names: names.append(k)
J = np.stack([B[n] for n in names]).astype(np.float32)
print(f"骨骼: {len(names)}根")

# 3.蒙皮权重(每顶点4骨骼,距离倒数^3)
d = np.linalg.norm(yup[:, None, :] - J[None, :, :], axis=2)
idx4 = np.argsort(d, axis=1)[:, :4]
w4 = 1.0 / (np.take_along_axis(d, idx4, 1) + 1e-4) ** 3
w4 = w4 / w4.sum(1, keepdims=True)
print(f"蒙皮: {w4.shape}")

# 4.walk动画(顶点序列→每骨骼每帧的位移+近似旋转)
walk = np.load(r"B:\mode\models\body3d\walk_cycle\walk12.npy")  # 12×13718×3 Z-up
walk_y = walk.copy()
walk_y[:, :, 1] = walk[:, :, 2]; walk_y[:, :, 2] = -walk[:, :, 1]
walk_y -= walk_y.reshape(-1, 3).mean(0)
walk_y[:, :, 1] -= walk_y[:, :, 1].min()
# 每帧骨骼位置(同一band逻辑对每帧跑太贵→用bindPose骨骼+顶点delta映射)
# 简化:骨骼动画=每帧骨骼的平移(取该骨骼邻近顶点的平均位移)
bind = J.copy()
def bone_delta(delta, j):
    near = np.argsort(np.linalg.norm(yup - J[j], axis=1))[:30]
    return delta[near].mean(0)
frames_j = []
for f in range(12):
    delta = walk_y[f] - yup
    jd = np.stack([bone_delta(delta, j) for j in range(len(names))])
    frames_j.append(bind + jd * 0.8)  # 0.8衰减防过冲
frames_j = np.stack(frames_j)  # 12×22×3
print(f"动画: 12帧×{len(names)}骨骼")

# 5.组装glTF
import struct
def pad(b): return b + b"\x00" * ((4 - len(b) % 4) % 4)
pos_bin = yup.astype(np.float32).tobytes()
idx_bin = F.astype(np.uint32).flatten().tobytes()  # flatten
jidx_bin = idx4.astype(np.uint16).tobytes()
jw_bin = w4.astype(np.float32).tobytes()
ibm = np.tile(np.eye(4, dtype=np.float32), (len(names), 1)).reshape(-1).tobytes()  # bind matrices单位阵(骨骼全局位)
jnt_bin = frames_j.astype(np.float32).tobytes()
time_bin = np.array([i / 24 for i in range(12)], np.float32).tobytes()
# buffers
blob = b""
def add(data):
    global blob
    off = len(blob); blob += pad(data); return off, len(data)
pos_off, pos_len = add(pos_bin)
idx_off, idx_len = add(np.array(F, np.uint32).tobytes())
jidx_off, jidx_len = add(jidx_bin)
jw_off, jw_len = add(jw_bin)
ibm_off, ibm_len = add(ibm)
jnt_off, jnt_len = add(jnt_bin)
time_off, time_len = add(time_bin)

g = GLTF2()
g.scene = 0
g.scenes = [Scene(nodes=[0, 1])]
# node0=mesh, node1+=骨骼(Hips为根,其余子级)
g.nodes = [Node(mesh=0, skin=0, name="Body")]
parent = {"Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1", "Neck": "Spine2", "Head": "Neck", "HeadTop_End": "Head",
          "LeftShoulder": "Spine2", "RightShoulder": "Spine2", "LeftArm": "LeftShoulder", "RightArm": "RightShoulder",
          "LeftForeArm": "LeftArm", "RightForeArm": "RightArm", "LeftHand": "LeftForeArm", "RightHand": "RightForeArm",
          "LeftUpLeg": "Hips", "RightUpLeg": "Hips", "LeftLeg": "LeftUpLeg", "RightLeg": "RightUpLeg",
          "LeftFoot": "LeftLeg", "RightFoot": "RightLeg", "LeftToeBase": "LeftFoot", "RightToeBase": "RightFoot"}
name2node = {}
for i, n in enumerate(names):
    name2node[n] = i + 1
for n in names:
    children = [name2node[c] for c, p in parent.items() if p == n]
    g.nodes.append(Node(name=n, children=children,
                        translation=[round(float(x), 5) for x in B[n]]))
g.nodes[name2node["Hips"]].children.append(0) if False else None
# skin
g.skins = [Skin(joints=list(range(1, len(names) + 1)),
                inverseBindMatrices=4,
                skeleton=name2node["Hips"])]
# mesh:attributes POSITION+JOINTS_0+WEIGHTS_0
g.meshes = [Mesh(primitives=[Primitive(attributes={"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3}, indices=1)])]
g.accessors = [
    Accessor(bufferView=0, componentType=5126, count=len(yup), type="VEC3", min=yup.min(0).tolist(), max=yup.max(0).tolist()),
    Accessor(bufferView=1, componentType=5125, count=len(F) * 3, type="SCALAR"),
    Accessor(bufferView=2, componentType=5123, count=len(yup), type="VEC4"),
    Accessor(bufferView=3, componentType=5126, count=len(yup), type="VEC4"),
    Accessor(bufferView=4, componentType=5126, count=len(names), type="MAT4"),
    Accessor(bufferView=5, componentType=5126, count=12 * len(names), type="VEC3"),
    Accessor(bufferView=6, componentType=5126, count=12, type="SCALAR"),
]
g.bufferViews = [
    BufferView(buffer=0, byteOffset=pos_off, byteLength=pos_len, target=34962),
    BufferView(buffer=0, byteOffset=idx_off, byteLength=idx_len, target=34963),
    BufferView(buffer=0, byteOffset=jidx_off, byteLength=jidx_len, target=34962),
    BufferView(buffer=0, byteOffset=jw_off, byteLength=jw_len, target=34962),
    BufferView(buffer=0, byteOffset=ibm_off, byteLength=ibm_len),
    BufferView(buffer=0, byteOffset=jnt_off, byteLength=jnt_len),
    BufferView(buffer=0, byteOffset=time_off, byteLength=time_len),
]
# animation: walk(translation轨道)
g.animations = [Animation(name="walk",
    samplers=[
        AnimationSampler(input=6, output=5, interpolation="LINEAR"),
    ],
    channels=[
        AnimationChannel(sampler=0, target=AnimationChannelTarget(node=name2node["Hips"], path="translation")),
    ])]
g.buffers = [Buffer(byteLength=len(blob))]
g.set_binary_blob(blob)
g.save(r"B:\mode\models\body3d\avatar_rigged.glb")
import os
print("avatar_rigged.glb:", os.path.getsize(r"B:\mode\models\body3d\avatar_rigged.glb") // 1024, "KB")
print("骨骼", len(names), "| 蒙皮VEC4 | walk 12帧 | GLB2.0")
