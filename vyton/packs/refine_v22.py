# cut_v2.2:细化层=trimap引导matting(在v2.1 mask上)
# 粗mask(erode确定前景/dilate确定背景/中间=未知区)→未知区alpha求解
import numpy as np
from PIL import Image
import cv2, glob, os, json

pool = json.load(open(r"B:/mode/models/cut/eval_domain_pool.json"))
class BackboneV2:
    pass  # 结构在onnx,直接用onnx推理
import onnxruntime as ort
bbs = ort.InferenceSession(r"B:/mode/models/cut/pack/capability_cutout_v2/backbone320.onnx", providers=["CPUExecutionProvider"])
hds = ort.InferenceSession(r"B:/mode/models/cut/pack/capability_cutout_v2/head320.onnx", providers=["CPUExecutionProvider"])

def coarse_mask(im, size=320):
    x = np.asarray(im.resize((size, size)), np.float32).transpose(2,0,1)/255.0
    f = bbs.run(None, {"image": x[None]})[0]
    m = hds.run(None, {"feat": f})[0]
    m = 1/(1+np.exp(-m[0,0]))
    return np.asarray(Image.fromarray((m*255).astype(np.uint8)).resize(im.size, Image.BILINEAR), np.float32)/255.0

def refine(im, m):
    """trimap引导精修:未知带内用色彩采样alpha"""
    W, H = im.size
    a = np.asarray(im, np.float32)
    m8 = (m*255).astype(np.uint8)
    # trimap: erode5=前景255, dilate5其余, 中间128
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fg = cv2.erode(m8, k, iterations=4)
    bg = cv2.dilate(m8, k, iterations=4)
    trimap = np.full((H, W), 128, np.uint8)
    trimap[fg > 127] = 255
    trimap[bg < 128] = 0
    # 未知区:引导滤波(以原图为引导,粗mask为输入)+色彩Laplacian近似
    # 快速guided filter自实现(He et al.版,均值盒滤波)
    def box(img, r):
        return cv2.boxFilter(img, -1, (2*r+1, 2*r+1), borderType=cv2.BORDER_REFLECT)
    def gfilter(g, x, r=6, eps=1e-4):
        g = g.astype(np.float32); x = x.astype(np.float32)
        if g.ndim == 2: g = g[..., None]
        if x.ndim == 2: x = x[..., None]
        if x.shape[-1] == 1 and g.shape[-1] == 3:
            x = np.repeat(x, 3, axis=-1)
        mg, mx = box(g, r), box(x, r)
        mgx = box(g*x, r)
        cov = mgx - mg*mx
        vg = box(g*g, r) - mg*mg
        w = cov / (vg.sum(-1, keepdims=True) + eps)
        b = mx - (w*mg).sum(-1, keepdims=True)
        return (box(w*g, r).sum(-1) + box(b, r)[..., 0]).astype(np.float32)
    refined = gfilter(a, m8)
    out = np.where(trimap == 128, refined, trimap).astype(np.float32)
    return np.clip(out/255.0, 0, 1)

ious, ses = [], []
for p in pool[:30]:
    k = os.path.basename(p) + ".png"
    lab_p = r"B:/mode/models/cut/pseudo_labels/" + k
    if not os.path.exists(lab_p): continue
    im = Image.open(p).convert("RGB")
    m = coarse_mask(im)
    r = refine(im, m)
    lab = np.asarray(Image.open(lab_p).convert("L").resize(im.size), np.float32)/255.0
    pred = (r > 0.5).astype(np.float32)
    inter = (pred*lab).sum(); union = ((pred+lab) > 0).sum()
    ious.append(inter/max(union,1))
    gy, gx = np.gradient(r)
    ses.append(float(((np.abs(gy)+np.abs(gx)) > 0.08).mean()))
print(f"v2.2(matting细化): IoU={np.mean(ious):.3f} 软边={np.mean(ses):.1%} (n={len(ious)})")
print("V22_NUM_DONE")


# 证据图:v2.1粗 vs v2.2细化 对比(6域内样本)
import random, json as js
pool2 = js.load(open(r"B:/mode/models/cut/eval_domain_pool.json"))
random.seed(11)
picks = random.sample(pool2, 4)
rows = []
for p in picks:
    im = Image.open(p).convert("RGB")
    W, H = im.size
    m1 = coarse_mask(im)
    m2 = refine(im, m1)
    def cutimg(m):
        c = np.asarray(im, np.float32).copy()
        c[..., :3] *= m[..., None]
        return c.astype(np.uint8)
    th = np.concatenate([np.asarray(im.resize((256,256))),
                         (m1*255).astype(np.uint8)[..., None].repeat(3, 2),
                         cutimg(m2)], axis=1)
    rows.append(np.asarray(Image.fromarray(th).resize((768, 256))))
    rows.append(np.full((4, 768, 3), 255, np.uint8))
grid = np.concatenate(rows[:-1], axis=0)
Image.fromarray(grid).save(r"B:/mode/models/cut/v22_evidence.png")
print("V22_EVG_OK")
