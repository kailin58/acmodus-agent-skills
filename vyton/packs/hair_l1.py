# 箭3-L1:发丝区强化细化(两轮trimap+gfilter)
import numpy as np
from PIL import Image
import cv2, glob, os, json
import onnxruntime as ort

bbs = ort.InferenceSession(r"B:/mode/models/cut/pack/capability_cutout_v2/backbone320.onnx", providers=["CPUExecutionProvider"])
hds = ort.InferenceSession(r"B:/mode/models/cut/pack/capability_cutout_v2/head320.onnx", providers=["CPUExecutionProvider"])

def box(img, r):
    return cv2.boxFilter(img, -1, (2*r+1, 2*r+1), borderType=cv2.BORDER_REFLECT)

def gfilter(g, x, r=6, eps=1e-4):
    g = g.astype(np.float32); x = x.astype(np.float32)
    if g.ndim == 2: g = g[..., None]
    if x.ndim == 2: x = x[..., None]
    if x.shape[-1] == 1 and g.shape[-1] == 3:
        x = np.repeat(x, 3, axis=-1)
    mg, mx = box(g, r), box(x, r)
    cov = box(g*x, r) - mg*mx
    vg = box(g*g, r) - mg*mg
    w = cov / (vg.sum(-1, keepdims=True) + eps)
    b = mx - (w*mg).sum(-1, keepdims=True)
    return (box(w*g, r).sum(-1) + box(b, r)[..., 0]).astype(np.float32)

def coarse(im, size=320):
    x = np.asarray(im.resize((size, size)), np.float32).transpose(2,0,1)/255.0
    f = bbs.run(None, {"image": x[None]})[0]
    m = hds.run(None, {"feat": f})[0]
    m = 1/(1+np.exp(-m[0,0]))
    return np.asarray(Image.fromarray((m*255).astype(np.uint8)).resize(im.size, Image.BILINEAR), np.float32)/255.0

def hair_region(m):
    H, W = m.shape
    m8 = (m*255).astype(np.uint8)
    edge = cv2.Canny(m8, 50, 150).astype(np.float32)
    edge = cv2.GaussianBlur(edge, (0,0), 5)
    top_w = np.clip(1.0 - np.arange(H, dtype=np.float32)/H*1.6, 0, 1)[:, None]
    hair = edge * top_w
    return (hair > 0.08).astype(np.float32)

def refine2(im, m):
    W, H = im.size
    a = np.asarray(im, np.float32)
    m8 = (m*255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fg = cv2.erode(m8, k, iterations=4); bg = cv2.dilate(m8, k, iterations=4)
    tri = np.full((H, W), 128, np.uint8); tri[fg>127]=255; tri[bg<128]=0
    r1 = gfilter(a, m8)
    out1 = np.where(tri==128, r1, tri.astype(np.float32))/255.0
    hr = hair_region(out1)
    m8b = (out1*255).astype(np.uint8)
    fg2 = cv2.erode(m8b, k, iterations=2); bg2 = cv2.dilate(m8b, k, iterations=2)
    tri2 = np.full((H, W), 128, np.uint8); tri2[fg2>127]=255; tri2[bg2<128]=0
    r2 = gfilter(a, m8b, r=4, eps=5e-4)
    out2 = np.where(tri2==128, r2, tri2.astype(np.float32))/255.0
    hr3 = cv2.GaussianBlur(hr, (0,0), 3)
    out = out1*(1-hr3*0.6) + out2*(hr3*0.6)
    return np.clip(out, 0, 1)

pool = json.load(open(r"B:/mode/models/cut/eval_domain_pool.json"))
import random
random.seed(4)
picks = [p for p in pool if "outfit" in p or "yt" in p][:6]
rows = []; ious = []
for p in picks:
    im = Image.open(p).convert("RGB")
    m = coarse(im)
    r = refine2(im, m)
    lab_p = r"B:/mode/models/cut/pseudo_labels/" + os.path.basename(p) + ".png"
    if os.path.exists(lab_p):
        lab = np.asarray(Image.open(lab_p).convert("L").resize(im.size), np.float32)/255.0
        pred = (r>0.5).astype(np.float32)
        inter = (pred*lab).sum(); union = ((pred+lab)>0).sum()
        ious.append(inter/max(union,1))
    c = np.asarray(im, np.float32).copy()
    c[..., :3] *= r[..., None]
    row = np.concatenate([np.asarray(im.resize((256,256))),
                          np.asarray(Image.fromarray((r*255).astype(np.uint8)).resize((256,256)))[..., None].repeat(3,2),
                          np.asarray(Image.fromarray(c.astype(np.uint8)).resize((256,256)))], axis=1)
    rows.append(row)
grid = np.concatenate(rows, axis=0)
Image.fromarray(grid).save(r"B:/mode/models/cut/v23_hair_evidence.png")
print(f"IOU={np.mean(ious):.3f}" if ious else "no-iou")
print("HAIR_L1_DONE")
