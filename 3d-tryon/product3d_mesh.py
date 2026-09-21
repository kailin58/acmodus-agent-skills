
# 续:白底图→深度→mesh(5张)
import os, hashlib
import numpy as np
from PIL import Image
import onnxruntime as ort

OUT = r"B:\mode\vton_training\garment_poc"
sess = ort.InferenceSession(r"B:\mode\models\depth_v2_small_fp32.onnx", providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name

def depth_of(img):
    x = np.array(img.resize((392, 522), Image.LANCZOS)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    x = ((x - mean) / std).transpose(2, 0, 1)[None]
    out = sess.run(None, {iname: x})[0].squeeze()
    d = 1.0 / (out + 1e-8)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return d

for f in sorted(os.listdir(OUT)):
    if not f.startswith("wb_"): continue
    wb = Image.open(os.path.join(OUT, f)).convert("RGB")
    d = depth_of(wb)
    g = np.array(wb.resize((392, 522), Image.LANCZOS).convert("L"))
    mask = g < 235
    ys, xs = np.where(mask)
    if len(ys) == 0: continue
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    d_c, m_c = d[y0:y1+1, x0:x1+1], mask[y0:y1+1, x0:x1+1]
    H, W = d_c.shape
    thick = max(H, W) * 0.15
    verts, vidx, vi = [], {}, 0
    for yy in range(H):
        for xx in range(W):
            if m_c[yy, xx]:
                vidx[(yy, xx)] = vi; vi += 1
                verts.append((xx, H - yy, float(d_c[yy, xx]) * thick))
    faces = []
    for yy in range(H - 1):
        for xx in range(W - 1):
            if all(k in vidx for k in [(yy,xx),(yy,xx+1),(yy+1,xx),(yy+1,xx+1)]):
                a,b,c,e = vidx[(yy,xx)], vidx[(yy,xx+1)], vidx[(yy+1,xx)], vidx[(yy+1,xx+1)]
                faces += [(a,c,b),(b,c,e)]
    obj = os.path.join(OUT, "mesh_" + f[3:-4] + ".obj")
    with open(obj, "w") as fh:
        for v in verts: fh.write(f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}\n")
        for fc in faces: fh.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")
    hh = hashlib.md5(open(obj, "rb").read()).hexdigest()[:10]
    print(f"MESH {f[3:20]}: {len(verts)}v {len(faces)}f {hh}", flush=True)
print("ALL_MESH_DONE", flush=True)
