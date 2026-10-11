
import json
import numpy as np
import trimesh
vs, fs = [], []
for line in open(r"B:\mode\vton_training\mesh_50\sample_0000.obj", errors="replace"):
    if line.startswith("v "):
        _, a, b, c = line.split(); vs.append([float(a), float(b), float(c)])
    elif line.startswith("f "):
        import re as _re
        idx = []
        for p in line.split()[1:]:
            m = _re.search(r"tensor\((\d+)\)", p)
            if m:
                idx.append(int(m.group(1)))   # tensor(N)→N(1-based)
            else:
                try:
                    idx.append(int(p.split("/")[0]))  # 正常格式
                except ValueError:
                    idx = []
                    break
        idx = [i - 1 for i in idx]
        if not idx or max(idx) >= len(vs) or min(idx) < 0:
            continue
        for k in range(1, len(idx) - 1):
            fs.append([idx[0], idx[k], idx[k + 1]])
v = np.array(vs, np.float32)
yup = v.copy(); yup[:, 1] = v[:, 2]; yup[:, 2] = -v[:, 1]
m = trimesh.Trimesh(vertices=yup, faces=np.array(fs), process=False)
m.visual.vertex_colors = [200, 180, 165, 255]  # 基础肤色
scene = trimesh.Scene()
scene.add_geometry(m, node_name="Body")
ok = scene.export(r"B:\mode\models\body3d\avatar_static.glb")
import os
print("GLB KB:", os.path.getsize(r"B:\mode\models\body3d\avatar_static.glb")//1024)
