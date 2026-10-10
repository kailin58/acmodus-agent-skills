"""vyton.core.metrics — 验收指标库(唯一实现)
lip_redness/skin_mask/texture_var/diff_region:全项目共用
"""
import numpy as np
from PIL import Image
from scipy import ndimage as nd


def lip_redness(img_arr):
    """唇部红度(化妆验收)"""
    H, W = img_arr.shape[:2]
    lip = img_arr[int(H*0.55):int(H*0.65), int(W*0.40):int(W*0.60)]
    return (lip[..., 0] - (lip[..., 1] + lip[..., 2]) / 2).mean()


def texture_var(gray_arr, region=None):
    """拉普拉斯纹理方差(毛孔/麻花验收)"""
    lap = nd.laplace(gray_arr.astype(np.float32))
    return lap[region].var() if region else lap.var()


def skin_mask(rgb_arr):
    """皮肤mask(HSV判据)"""
    hsv = np.asarray(Image.fromarray(rgb_arr).convert("HSV"))
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return ((h < 28) | (h > 235)) & (s > 15) & (s < 200) & (v > 50)


def face_ellipse(H, W, cy=0.32, ey=0.30, cx=0.5, ex=0.36, bbox=None):
    """脸区椭圆mask;bbox=(x1,y1,x2,y2)可选=检测器输出优先"""
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cx_, ex_ = (x1+x2)/2/W, max((x2-x1)/2, 8)/W
        cy_, ey_ = (y1+y2)/2/H, max((y2-y1)/2, 8)/H
        cy, ey, cx, ex = cy_, ey_, cx_, ex_
    yy, xx = np.mgrid[0:H, 0:W]
    return (((yy - H*cy)/(H*ey))**2 + ((xx - W*cx)/(W*ex))**2) < 1.0
