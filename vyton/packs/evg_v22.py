# v2.2证据图独立跑
import numpy as np
from PIL import Image
import cv2, os, json, random
import onnxruntime as ort
bbs = ort.InferenceSession(r"B:/mode/models/cut/pack/capability_cutout_v2/backbone320.onnx", providers=["CPUExecutionProvider"])
hds = ort.InferenceSession(r"B:/mode/models/cut/pack/capability_cutout_v2/head320.onnx", providers=["CPUExecutionProvider"])
def coarse_mask(im, size=320):
    x = np.asarray(im.resize((size, size)), np.float32).transpose(2,0,1)/255.0
    f = bbs.run(None, {"image": x[None]})[0]
    m = hds.run(None, {"feat": f})[0]
    m = 1/(1+np.exp(-m[0,0]))
    return np.asarray(Image.fromarray((m*255).astype(np.uint8)).resize(im.size, Image.BILINEAR), np.float32)/255.0
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
def refine(im, m):
    W, H = im.size
    a = np.asarray(im, np.float32)
    m8 = (m*255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fg = cv2.erode(m8, k, iterations=4)
    bg = cv2.dilate(m8, k, iterations=4)
    trimap = np.full((H, W), 128, np.uint8)
    trimap[fg > 127] = 255
    trimap[bg < 128] = 0
    refined = gfilter(a, m8)
    out = np.where(trimap == 128, refined, trimap.astype(np.float32))
    return np.clip(out/255.0, 0, 1)
pool = json.load(open(r"B:/mode/models/cut/eval_domain_pool.json"))
random.seed(11)
picks = random.sample(pool, 4)
rows = []
for p in picks:
    im = Image.open(p).convert("RGB")
    m1 = coarse_mask(im)
    m2 = refine(im, m1)
    c = np.asarray(im, np.float32).copy()
    c[..., :3] *= m2[..., None]
    row = np.concatenate([
        np.asarray(im.resize((256,256))),
        np.asarray(Image.fromarray((m1*255).astype(np.uint8)).resize((256,256)))[..., None].repeat(3,2),
        np.asarray(Image.fromarray(c.astype(np.uint8)).resize((256,256)))], axis=1)
    rows.append(row)
    rows.append(np.full((4, 768, 3), 255, np.uint8))
grid = np.concatenate(rows[:-1], axis=0)
Image.fromarray(grid).save(r"B:/mode/models/cut/v22_evidence.png")
print("V22_EVG_OK", grid.shape)
