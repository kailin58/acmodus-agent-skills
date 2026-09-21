
# 批量商品3D化:53张Pinterest商品全量
import os, hashlib
import numpy as np
from PIL import Image
from rembg import remove
import onnxruntime as ort

SRC = r"B:\mode\pinterest\images"
OUT = r"B:\mode\product3d"
os.makedirs(os.path.join(OUT, "white"), exist_ok=True)
os.makedirs(os.path.join(OUT, "mesh"), exist_ok=True)

sess = ort.InferenceSession(r"B:\mode\models\depth_v2_small_fp32.onnx", providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name

def depth_of(img):
    x = np.array(img.resize((392, 522), Image.LANCZOS)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    x = ((x - mean) / std).transpose(2, 0, 1)[None]
    out = sess.run(None, {iname: x})[0].squeeze()
    d = 1.0 / (out + 1e-8)
    return (d - d.min()) / (d.max() - d.min() + 1e-8)

done = set(os.listdir(os.path.join(OUT, "mesh")))
files = [f for f in sorted(os.listdir(SRC)) if f.lower().endswith((".jpg",".png",".webp"))]
print(f"总数{len(files)} 已完成{len(done)}", flush=True)
for f in files:
    objn = "mesh_" + os.path.splitext(f)[0] + ".obj"
    if objn in done: continue
    try:
        img = Image.open(os.path.join(SRC, f)).convert("RGB")
        img.thumbnail((512, 512))
        cut = remove(img)
        bg = Image.new("RGB", cut.size, (255,255,255))
        bg.paste(cut, mask=cut.split()[3] if cut.mode=="RGBA" else None)
        bg.save(os.path.join(OUT, "white", os.path.splitext(f)[0] + ".png"))
        d = depth_of(bg)
        g = np.array(bg.resize((392,522), Image.LANCZOS).convert("L"))
        mask = g < 235
        ys, xs = np.where(mask)
        if len(ys)==0: print("EMPTY", f, flush=True); continue
        y0,y1,x0,x1 = ys.min(),ys.max(),xs.min(),xs.max()
        d_c,m_c = d[y0:y1+1,x0:x1+1], mask[y0:y1+1,x0:x1+1]
        H,W = d_c.shape
        thick = max(H,W)*0.15
        verts,vidx,vi = [],{},0
        for yy in range(H):
            for xx in range(W):
                if m_c[yy,xx]:
                    vidx[(yy,xx)]=vi; vi+=1
                    verts.append((xx,H-yy,float(d_c[yy,xx])*thick))
        faces=[]
        for yy in range(H-1):
            for xx in range(W-1):
                if all(k in vidx for k in [(yy,xx),(yy,xx+1),(yy+1,xx),(yy+1,xx+1)]):
                    a,b,c,e = vidx[(yy,xx)],vidx[(yy,xx+1)],vidx[(yy+1,xx)],vidx[(yy+1,xx+1)]
                    faces += [(a,c,b),(b,c,e)]
        obj = os.path.join(OUT,"mesh",objn)
        with open(obj,"w") as fh:
            for v in verts: fh.write(f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}\n")
            for fc in faces: fh.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")
        print("OK", f[:16], len(verts), flush=True)
    except Exception as ex:
        print("ERR", f[:16], str(ex)[:50], flush=True)
print("BATCH_DONE", flush=True)
