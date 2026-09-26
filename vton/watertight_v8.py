# 水密mesh v8:三次样条轮廓(终极光滑)
import numpy as np
from PIL import Image, ImageDraw
import glob, math

white = Image.open(sorted(glob.glob(r"B:\mode\vton_training\whitebg_50\sample_0000.jpg"))[0]).convert("RGB")
wa = np.array(white, np.float32)
mask = (wa.sum(2) < 720)
step = 4
m = mask[::step, ::step]
h2, w2 = m.shape

raw = []
for j in range(h2):
    xs = np.where(m[j])[0]
    if len(xs) >= 2:
        t = j/h2
        cx = (xs.min()+xs.max())/2; hw = (xs.max()-xs.min())/2+1
        ratio = 0.55 if t<0.35 else (0.45 if t<0.5 else (0.62 if t<0.62 else 0.42))
        raw.append((j, cx, hw, ratio))
    else:
        raw.append((j, np.nan, np.nan, np.nan))
arr = np.array([[r[1],r[2],r[3]] for r in raw])
for c in range(3):
    s = arr[:,c]; idx = np.where(~np.isnan(s))[0]
    arr[:,c] = np.interp(np.arange(len(s)), idx, s[idx])

# 三次样条重采(60个控制行,数学光滑)
from numpy.polynomial import polynomial as P
jj = np.arange(h2)
ctrl = np.linspace(0, h2-1, 60).astype(int)
def spline_resample(y_vals, ctrl_x, n_out):
    # 分段三次插值(光滑C1)
    out = np.zeros(n_out)
    xs = np.linspace(0, ctrl_x[-1], n_out)
    # 用numpy interp + 低次多项式平滑近似:直接3次样条
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(ctrl_x, y_vals[ctrl_x], bc_type="natural")
        return cs(xs)
    except ImportError:
        # 无scipy:滑动大窗高斯
        k = 25
        ker = np.exp(-0.5*(np.arange(k)-k//2)**2/(k/4)**2); ker/=ker.sum()
        return np.convolve(np.pad(y_vals,(k//2,k//2),mode="edge"), ker, mode="valid")
n_rows = 120
jj_new = np.linspace(0, h2-1, n_rows)
cx_s = spline_resample(arr[:,0], ctrl, n_rows)
hw_s = spline_resample(arr[:,1], ctrl, n_rows)
rt_s = spline_resample(arr[:,2], ctrl, n_rows)

verts, faces = [], []
NS = 24
rings = []
for ri in range(n_rows):
    j = jj_new[ri]
    y = 0.5 - j/(h2-1); x0 = cx_s[ri]/(w2-1)-0.5; r = hw_s[ri]/(w2-1); ra = rt_s[ri]
    ids = []
    for k in range(NS):
        th = 2*math.pi*k/NS
        ids.append(len(verts))
        verts.append((x0+r*math.cos(th), y, r*ra*math.sin(th)))
    rings.append(ids)
for ri in range(n_rows-1):
    a, b = rings[ri], rings[ri+1]
    for k in range(NS):
        k2 = (k+1)%NS
        faces.append((a[k], b[k], b[k2])); faces.append((a[k], b[k2], a[k2]))
# 封顶底
c0 = len(verts); verts.append((0.0, verts[rings[0][0]][1], 0.0))
for k in range(NS): faces.append((c0, rings[0][k], rings[0][(k+1)%NS]))
c1 = len(verts); verts.append((0.0, verts[rings[-1][0]][1], 0.0))
for k in range(NS): faces.append((c1, rings[-1][(k+1)%NS], rings[-1][k]))

with open(r"B:\mode\vton_training\mesh_50\sample_0000_watertight_v8.obj", "w") as f:
    for v in verts: f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
    for a,b,c in faces: f.write(f"f {a+1} {b+1} {c+1}\n")
V = np.array(verts); F = np.array(faces)
print("V8_DONE", len(V), "verts", len(F), "faces")

# 顶点法线+柔光渲染(spec弱化)
VN = np.zeros_like(V)
fn = np.cross(V[F[:,1]]-V[F[:,0]], V[F[:,2]]-V[F[:,0]]); fn = fn/(np.linalg.norm(fn,axis=1,keepdims=True)+1e-9)
for i in range(3): np.add.at(VN, F[:,i], fn)
VN = VN/(np.linalg.norm(VN,axis=1,keepdims=True)+1e-9)
LIGHT = np.array([0.45,0.55,0.70]); LIGHT/=np.linalg.norm(LIGHT)
def render(view, W2=320, H2=380, scale=3):
    img = Image.new("RGB", (W2*scale,H2*scale),(16,16,22)); dr = ImageDraw.Draw(img)
    ctr = V[F].mean(1)
    if view=="front": Vs=V; VNs=VN; key=ctr[:,2]; ax1=0
    elif view=="side": Vs=V; VNs=VN; key=-np.abs(ctr[:,0]); ax1=2
    else:
        ang=math.radians(35); rot=np.array([[math.cos(ang),0,-math.sin(ang)],[0,1,0],[math.sin(ang),0,math.cos(ang)]])
        Vs=V@rot.T; VNs=VN@rot.T; key=(Vs[F].mean(1))[:,2]; ax1=0
    order=np.argsort(key)
    fv=VNs[F].mean(1); fv=fv/(np.linalg.norm(fv,axis=1,keepdims=True)+1e-9)
    diffuse=np.clip(fv@LIGHT,0,1)
    facing=fv@np.array([0,0,1.0])
    rim=np.clip(1-np.abs(facing),0,1)**2*0.4
    spec=np.clip(fv@LIGHT,0,1)**32*0.3
    for fi in order:
        f3=F[fi]
        pts=[(int((Vs[i][ax1]+0.45)*(W2-40)*scale), int((0.52-Vs[i][1])*(H2-50)*scale)) for i in f3]
        b=0.15+0.80*diffuse[fi]+rim[fi]+spec[fi]
        c=int(255*min(b,1.02)**0.95)
        dr.polygon(pts, fill=(int(c*0.82),int(c*0.92),int(c*0.88)))
    return img.resize((W2,H2), Image.LANCZOS)
canvas = Image.new("RGB", (1000,400),(14,14,20))
for k2, vw in enumerate(["front","side","3q"]):
    canvas.paste(render(vw), (12+k2*330, 10))
ImageDraw.Draw(canvas).text((12,12), "v8 cubic-spline contour (C1 smooth)", fill=(255,255,60))
canvas.save(r"B:\mode\vton_training\mesh_50\watertight_v8_preview.png")
print("PREVIEW_DONE")
