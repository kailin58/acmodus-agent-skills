"""vyton.body3d.face — 数据+脸照→有你脸的虚拟自己
apply_face(mesh, face_img):脸照提取(脸宽比/下巴形状/肤色RGB)→头部网格变形+着色
端侧:纯numpy+PIL,无重依赖
"""
import numpy as np


def _face_box(img_arr):
    """极简脸定位:肤色检测(椭圆RGB聚类)→包围盒;失败返回None"""
    rgb = img_arr.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # YCbCr肤色判别(Cb 77-127, Cr 133-173)
    cb = 128 - 0.168736*r - 0.331264*g + 0.5*b
    cr = 128 + 0.5*r - 0.418688*g - 0.081312*b
    mask = (cb > 77) & (cb < 127) & (cr > 133) & (cr < 173)
    ys, xs = np.where(mask)
    if len(xs) < 500:
        return None
    x1, x2 = np.percentile(xs, [5, 95])
    y1, y2 = np.percentile(ys, [5, 95])
    sel = mask[int(y1):int(y2), int(x1):int(x2)]
    return dict(x1=float(x1), x2=float(x2), y1=float(y1), y2=float(y2),
                skin_rgb=rgb[int(y1):int(y2), int(x1):int(x2)][sel].mean(0) / 255.0)


def apply_face(mesh, face_img_arr, strength=1.0):
    """脸照→头部变形+肤色。mesh: Nx3, face_img_arr: HxWx3 uint8
    返回(mesh, info): 脸宽/下巴调整+肤色着色数组"""
    info = _face_box(face_img_arr)
    if info is None:
        return mesh, None
    fw = (info["x2"] - info["x1"]) / face_img_arr.shape[1]       # 脸宽占图比
    fh = (info["y2"] - info["y1"]) / face_img_arr.shape[0]
    aspect = (info["x2"] - info["x1"]) / max(info["y2"] - info["y1"], 1)  # 宽高比(圆脸/长脸)
    # 网格头区
    z = mesh[:, 2]; H = z.max() - z.min()
    h = (z - z.min()) / H
    head_i = h > 0.86
    head = mesh[head_i]
    hh = (head[:, 2] - head[:, 2].min()) / (head[:, 2].max() - head[:, 2].min())
    # 1.脸宽高比→头横向缩放(圆脸aspect>0.75→宽,长脸<0.65→窄)
    f = np.clip((aspect - 0.70) / 0.20, -1, 1) * 0.18 * strength
    wf = 1 + f * (1 - np.abs(hh - 0.45) * 1.2).clip(0)           # 头中部变化最大
    head[:, 0] *= wf
    # 2.下巴(头下缘0-0.25)窄化=尖下巴;宽=圆下巴(与aspect联动)
    jaw = np.clip((0.25 - hh) / 0.25, 0, 1)
    head[:, 0] *= (1 + f * 0.5 * jaw)
    mesh = mesh.copy()
    mesh[head_i] = head
    return mesh, dict(face_width_ratio=fw, face_aspect=aspect,
                      skin_rgb=info["skin_rgb"].tolist(), applied=True)
