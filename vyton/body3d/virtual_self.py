"""vyton.body3d.virtual_self — 统一入口(4输入形式)
形式①照片三视图+身高体重(调measure模块)
形式②基础数据(身高体重年龄性别国家民族)
形式③明细数据(+围度)
形式④数据+面部照片
规则(用户定):数据越细越接近;不全→模拟完善(complete_params)+inferred报告
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from body3d.complete import complete_params
from body3d.avatar import gen_avatar

_BASE = None
def _base_mesh():
    global _BASE
    if _BASE is None:
        def read_obj_v(p):
            vs = []
            for line in open(p, errors="replace"):
                if line.startswith("v "):
                    _, a, b, c = line.split()
                    vs.append([float(a), float(b), float(c)])
            return np.array(vs, np.float32)
        _BASE = read_obj_v(r"B:\mode\vton_training\mesh_50\sample_0000.obj")
    return _BASE

def gen_virtual_self(age=None, sex=None, height=None, weight=None,
                     ethnic=None, country=None, bust=None, waist=None, hips=None,
                     face_img=None, face_strength=1.0):
    """任意输入形式→虚拟自己。返回(mesh, report)"""
    p, inf = complete_params(age, sex, height, weight, ethnic, country, bust, waist, hips)
    mesh, meta = gen_avatar(_base_mesh(), p["age"], p["sex"], p["height"], p["weight"], p["ethnic"])
    report = dict(params=p, inferred=inf,
                  precision="高" if len(inf) == 0 else ("中" if len(inf) <= 3 else "低"),
                  note="数据越细越接近;推算字段:" + (",".join(inf) if inf else "无"))
    if face_img is not None:
        from body3d.face import apply_face
        mesh, finfo = apply_face(mesh, face_img, strength=face_strength)
        report["face"] = finfo
    return mesh, report
