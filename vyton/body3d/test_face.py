
import sys
sys.path.insert(0, r"B:\\mode\\vyton")
import numpy as np
from body3d.face import apply_face
from body3d.avatar import gen_avatar
def read_obj_v(p):
    vs = []
    for line in open(p, errors="replace"):
        if line.startswith("v "):
            _, a, b, c = line.split()
            vs.append([float(a), float(b), float(c)])
    return np.array(vs, np.float32)
base = read_obj_v(r"B:\\mode\\vton_training\\mesh_50\\sample_0000.obj")
v, meta = gen_avatar(base, 28, "m", 175, 70)
# 造两张假脸照:圆脸(宽椭圆)vs长脸(竖椭圆)
from PIL import Image, ImageDraw
def fake_face(w_ratio):
    im = Image.new("RGB", (200, 260), (40, 40, 50))
    d = ImageDraw.Draw(im)
    cx, cy = 100, 130
    rw, rh = int(90*w_ratio), 110
    d.ellipse([cx-rw, cy-rh, cx+rw, cy+rh], fill=(224, 172, 140))
    return np.asarray(im)
round_img = fake_face(1.0)   # 圆脸
long_img = fake_face(0.62)   # 长脸
v1, i1 = apply_face(v, round_img)
v2, i2 = apply_face(v, long_img)
print("圆脸:", i1 and round(i1["face_aspect"],2), "长脸:", i2 and round(i2["face_aspect"],2))
hw1 = np.ptp(v1[v1[:,2] > v1[:,2].min() + 0.9*(v1[:,2].max()-v1[:,2].min()), 0])
hw2 = np.ptp(v2[v2[:,2] > v2[:,2].min() + 0.9*(v2[:,2].max()-v2[:,2].min()), 0])
print(f"圆脸头宽{hw1*100:.1f}cm vs 长脸头宽{hw2*100:.1f}cm (应圆>长)")
assert hw1 > hw2
print("PASS: 脸照→头型变形生效")
