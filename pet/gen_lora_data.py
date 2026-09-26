# 宠物穿衣LoRA数据合成:50对(原图→程序穿衣图)
import io, os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import onnxruntime as ort

OUT = r"B:\mode\pet\lora_data"
os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet(r"B:\mode\pet\breed_data\train.parquet")
dogs = df[df["label_cat_dog"]==1].reset_index(drop=True)
sess = ort.InferenceSession(r"B:\mode\models\pet\rtmpose_animal_ap10k.onnx", providers=["CPUExecutionProvider"])

def smart_crop(im):
    a = np.asarray(im.convert("RGB").resize((128,128)), np.float32)
    lum = a.mean(2)
    border = np.concatenate([lum[0], lum[-1], lum[:,0], lum[:,-1]])
    bg = np.median(border)
    fg = np.abs(lum - bg) > 24
    if fg.sum() < 200: return None
    ys, xs = np.where(fg)
    w0, h0 = im.size
    c = im.crop((max(0,int(xs.min()/128*w0)-8), max(0,int(ys.min()/128*h0)-8),
                 min(w0,int(xs.max()/128*w0)+8), min(h0,int(ys.max()/128*h0)+8)))
    w2, h2 = c.size
    s2 = min(w2, h2)
    return c.crop(((w2-s2)//2,(h2-s2)//2,(w2+s2)//2,(h2+s2)//2))

def kps_of(im256):
    x = np.asarray(im256, np.float32)
    x = ((x-np.array([123.675,116.28,103.53],np.float32))/np.array([58.395,57.12,57.375],np.float32)).transpose(2,0,1)[None]
    oo = sess.run(None, {sess.get_inputs()[0].name: x})
    kk = np.stack([oo[0][0].argmax(-1)/2, oo[1][0].argmax(-1)/2], 1)
    return kk, oo[0][0].max(-1)

import random
random.seed(7)
COLORS = [(62,48,"red"),(30,70,"blue"),(45,55,"green"),(75,40,"yellow"),(20,25,"black")]
n_ok = 0
caps = open(os.path.join(OUT, "captions.tsv"), "w", encoding="utf-8")
for idx in range(len(dogs)):
    if n_ok >= 50: break
    row = dogs.iloc[idx]
    im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    cr = smart_crop(im)
    if cr is None: continue
    im512 = cr.resize((512,512))
    kk, cc = kps_of(im512.resize((256,256)))
    if cc[4] < 0.5 or cc[5] < 0.5: continue
    sc = 2.0
    neck, ls, rs, lh, rh = kk[4]*sc, kk[5]*sc, kk[8]*sc, kk[11]*sc, kk[14]*sc
    W = H = 512
    y_top = min(ls[1], rs[1]) + 14
    y_bot = max(lh[1], rh[1]) + 16
    if y_bot - y_top < 60: continue
    def bc(t):
        cx = (ls[0]+rs[0])/2*(1-t)+(lh[0]+rh[0])/2*t
        return np.array([cx, y_top*(1-t)+y_bot*t])
    def bh(t):
        return max(np.linalg.norm(ls-rs)/2*1.1, 24)*(1-t)+max(np.linalg.norm(lh-rh)/2*1.2, 28)*t
    mask = Image.new("L",(W,H),0)
    dm = ImageDraw.Draw(mask)
    for r in range(60):
        t = r/59
        c = bc(t); hw = bh(t)*1.06
        dm.ellipse((c[0]-hw, c[1]-9, c[0]+hw, c[1]+9), fill=255)
    # 头保护
    hx, hy = (kk[0][0]+kk[1][0])/2*sc, (kk[0][1]+kk[1][1])/2*sc
    head_r = max(np.linalg.norm(kk[4]*sc-[hx,hy])*1.6, 70)
    mh = Image.new("L",(W,H),255)
    ImageDraw.Draw(mh).ellipse((hx-head_r,hy-head_r,hx+head_r,hy+head_r),fill=0)
    mh = mh.filter(ImageFilter.GaussianBlur(10))
    marr = np.minimum(np.asarray(mask,np.float32), np.asarray(mh,np.float32))
    mask = Image.fromarray(marr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(7))
    # Lab染色(随机色)
    ta, tb, cname = random.choice(COLORS)
    src_lab = np.array(im512.convert("LAB"), np.float32)
    m = np.asarray(mask, np.float32)[...,None]/255.0
    xs2 = np.linspace(-1,1,W)[None,:]
    lm = 1.0 + 0.18*(-xs2)
    nl = src_lab.copy()
    nl[...,1] = src_lab[...,1]*(1-m[...,0]) + ta*lm*m[...,0]
    nl[...,2] = src_lab[...,2]*(1-m[...,0]) + tb*lm*m[...,0]
    worn = Image.fromarray(np.clip(nl,0,255).astype(np.uint8), mode="LAB").convert("RGB")
    base = f"dog_{n_ok:03d}"
    im512.save(os.path.join(OUT, base+"_src.png"))
    worn.save(os.path.join(OUT, base+"_worn.png"))
    caps.write(f"{base}_worn.png\ta dog wearing a {cname} knitted sweater\n")
    n_ok += 1
caps.close()
print("LORA_DATA_DONE", n_ok, flush=True)
