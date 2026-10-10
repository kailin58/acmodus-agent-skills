
import sys
sys.path.insert(0, r"B:\\mode\\vyton")
import numpy as np
from body3d.avatar import gen_avatar, stage_of
# 基础网格(模块自带读法)
def read_obj_v(p):
    vs = []
    for line in open(p, errors="replace"):
        if line.startswith("v "):
            _, a, b, c = line.split()
            vs.append([float(a), float(b), float(c)])
    return np.array(vs, np.float32)
base = read_obj_v(r"B:\\mode\\vton_training\\mesh_50\\sample_0000.obj")
ok = 0
# 1.身高精确性
for h in [50, 100, 145, 178, 195]:
    v, m = gen_avatar(base, age=30, sex="m", height=h, weight=70)
    H = (v[:,2].max()-v[:,2].min())*100
    assert abs(H-h) < 2, f"{h}→{H}"
    ok += 1
# 2.体重单调(越重越宽)
ws = []
for w in [50, 70, 100]:
    v, _ = gen_avatar(base, 30, "m", 178, w)
    ws.append(v[:,0].max()-v[:,0].min())
assert ws[0] < ws[1] < ws[2], ws
ok += 1
# 3.性别差(同龄同高,男肩>女肩)
vm, _ = gen_avatar(base, 25, "m", 170, 65)
vf, _ = gen_avatar(base, 25, "f", 170, 65)
def shoulder_w(v):
    mid = np.median(v[:,0]); H = v[:,2].max()-v[:,2].min()
    h = (v[:,2]-v[:,2].min())/H
    t = v[(h>0.80)&(h<0.88)&(np.abs(v[:,0]-mid)<0.12*H)]
    return t[:,0].max()-t[:,0].min()
assert shoulder_w(vm) > shoulder_w(vf)
ok += 1
# 4.年龄单调(婴儿头身比>成人)
vb, _ = gen_avatar(base, 0.5, "m", weight=8)
va, _ = gen_avatar(base, 30, "m", height=175)
def head_ratio(v):
    mid = np.median(v[:,0]); H = v[:,2].max()-v[:,2].min()
    h = (v[:,2]-v[:,2].min())/H
    hd = v[(h>0.90)&(np.abs(v[:,0]-mid)<0.4*H)]
    return (hd[:,0].max()-hd[:,0].min())/H
assert head_ratio(vb) > head_ratio(va)
ok += 1
# 5.同输入确定性
v1, _ = gen_avatar(base, 30, "f", 165, 55)
v2, _ = gen_avatar(base, 30, "f", 165, 55)
assert np.array_equal(v1, v2)
ok += 1
print(f"PASS {min(ok,5)}/5")
