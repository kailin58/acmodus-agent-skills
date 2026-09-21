
# 商品3D化POC:真实商品图(Pinterest前5张)→rembg→白底→深度→mesh
import os
import numpy as np
from PIL import Image

SRC = r"B:\mode\pinterest\images"
OUT = r"B:\mode\vton_training\garment_poc"
os.makedirs(OUT, exist_ok=True)

# ① rembg抠图→白底
from rembg import remove
files = sorted(os.listdir(SRC))[:5]
for f in files:
    img = Image.open(os.path.join(SRC, f)).convert("RGB")
    img.thumbnail((512, 512))
    cut = remove(img)  # RGBA
    # 白底合成
    bg = Image.new("RGB", cut.size, (255, 255, 255))
    bg.paste(cut, mask=cut.split()[3] if cut.mode == "RGBA" else None)
    bg.save(os.path.join(OUT, f"wb_{f}.png"))
    cut.save(os.path.join(OUT, f"cut_{f}.png"))
    print("WHITEBG", f, flush=True)

# ② 深度(Depth-Anything-V2)
import onnxruntime as ort
sess = ort.InferenceSession(r"B:\mode\models\depth_anything_v2_small.onnx", providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
def depth_of(img):
    w, h = img.size
    x = np.array(img.resize((392, 522), Image.LANCZOS)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    x = ((x - mean) / std).transpose(2, 0, 1)[None]
    out = sess.run(None, {iname: x.astype(np.float32)})[0].squeeze()
    d = 1.0 / (out + 1e-8)  # 逆深度→近似近大远小
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return d

# ③ mesh(高度场,前景mask,每图唯一)
for f in files:
    wb = Image.open(os.path.join(OUT, f"wb_{f}.png")).convert("RGB")
    d = depth_of(wb)
    g = np.array(ww := wb.resize((392, 522), Image.LANCZOS).convert("L"))
    mask = g < 235
    ys, xs = np.where(mask)
    if len(ys) == 0: print("EMPTY", f); continue
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
    obj = os.path.join(OUT, f"mesh_{f}.obj")
    with open(obj, "w") as fh:
        for v in verts: fh.write(f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}\n")
        for fc in faces: fh.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")
    # 唯一性哈希
    import hashlib
    hh = hashlib.md5(open(obj,"rb").read()).hexdigest()[:10]
    print(f"MESH {f}: {len(verts)}verts {len(faces)}faces hash={hh}", flush=True)
print("POC_DONE", flush=True)
