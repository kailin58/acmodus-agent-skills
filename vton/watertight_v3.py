# 水密mesh v3:椭圆截面体积法(侧视不再空心)
import numpy as np
from PIL import Image, ImageDraw
import glob, math

white = Image.open(sorted(glob.glob(r"B:\mode\vton_training\whitebg_50\sample_0000.jpg"))[0]).convert("RGB")
W, H = white.size
wa = np.array(white, np.float32)
mask = (wa.sum(2) < 720)

step = 6
m = mask[::step, ::step]
h2, w2 = m.shape

# 每行:服装左右边界→宽度→椭圆截面(厚度=宽度×0.35)
rows = []
for j in range(h2):
    xs = np.where(m[j])[0]
    if len(xs) >= 2:
        cx = (xs.min()+xs.max())/2
        hw = (xs.max()-xs.min())/2+1
        rows.append((j, cx, hw))
    else:
        rows.append((j, None, 0))

verts, faces = [], []
THICK = 0.35  # 椭圆短轴比
def ring(j, cx, hw):
    """每行生成12点椭圆环"""
    ids = []
    y = 0.5 - j/(h2-1)
    x0 = cx/(w2-1)-0.5
    r = hw/(w2-1)
    for k in range(12):
        th = 2*math.pi*k/12
        px = x0 + r*math.cos(th)
        pz = r*THICK*math.sin(th)
        ids.append(len(verts))
        verts.append((px, y, pz+r*THICK*0.3))
    return ids

prev = None
for j, cx, hw in rows:
    cur = ring(j, cx, hw) if cx is not None else None
    if prev and cur:
        for k in range(12):
            k2 = (k+1) % 12
            faces.append((prev[k], cur[k], cur[k2]))
            faces.append((prev[k], cur[k2], prev[k2]))
    prev = cur
# 封顶封底
first = ring(rows[0][0], rows[0][1], rows[0][2]) if rows[0][1] is not None else None
last_rows = [r for r in rows if r[1] is not None]
last = ring(last_rows[-1][0], last_rows[-1][1], last_rows[-1][2])
if first:
    c0 = len(verts); verts.append(((sum(verts[i][0] for i in first)/12), verts[first[0]][1], 0.05))
    for k in range(12):
        faces.append((c0, first[k], first[(k+1)%12]))
if last:
    c1 = len(verts); verts.append(((sum(verts[i][0] for i in last)/12), verts[last[0]][1], 0.05))
    for k in range(12):
        faces.append((c1, last[(k+1)%12], last[k]))

out = r"B:\mode\vton_training\mesh_50\sample_0000_watertight_v3.obj"
with open(out, "w") as f:
    for v in verts: f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
    for a,b,c in faces: f.write(f"f {a+1} {b+1} {c+1}\n")
print("V3_DONE verts:", len(verts), "faces:", len(faces))

# 三视角
def render(view, W2=280, H2=340):
    img = Image.new("RGB", (W2,H2), (16,16,22))
    dr = ImageDraw.Draw(img)
    for x,y,z in verts:
        if view=="front": px,py = x,y
        elif view=="side": px,py = (z)*1.6, y
        else: px,py = x*0.72 - z*0.55, y*0.88-0.04
        c = int(85+165*max(0,min(1,z*2.5)))
        dr.ellipse((int(px*(W2-40))+W2//2-1, int((py+0.5)*(H2-40))+20-1,
                    int(px*(W2-40))+W2//2+1, int((py+0.5)*(H2-40))+20+1), fill=(c,c+20,165))
    return img
canvas = Image.new("RGB", (880, 360), (14,14,20))
for k, vw in enumerate(["front","side","3q"]):
    canvas.paste(render(vw), (10+k*292, 10))
ImageDraw.Draw(canvas).text((12,12), "Watertight v3 (elliptical cross-sections): front/side/3Q", fill=(255,255,60))
canvas.save(r"B:\mode\vton_training\mesh_50\watertight_v3_preview.png")
print("PREVIEW_DONE")
