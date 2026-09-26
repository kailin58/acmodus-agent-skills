# 宠物穿衣LoRA数据合成:50对(原图→程序穿衣图)
import io, os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import onnxruntime as ort

OUT = r"B:\mode\pet\lora_data_v3"
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
    kk = np.stack([oo[0][0].argmax(-1)/2, oo[1][0].argmax(-1)/2], 1) * 2.0  # 512系
    return kk, oo[0][0].max(-1)

import random
random.seed(7)
COLORS = [(62,48,"red"),(30,70,"blue"),(45,55,"green"),(75,40,"yellow"),(20,25,"black")]
n_ok = 0
caps = open(os.path.join(OUT, "captions.tsv"), "w", encoding="utf-8")
for idx in range(len(dogs)):
    if n_ok >= 65: break
    row = dogs.iloc[idx]
    im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    cr = smart_crop(im)
    if cr is None: continue
    im512 = cr.resize((512,512))
    kk, cc = kps_of(im512.resize((256,256)))
    if cc[4] < 0.5 or cc[5] < 0.5: continue
    neck, ls, rs, lh, rh = kk[4], kk[5], kk[8], kk[11], kk[14]
    W = H = 512
    y_top = max(min(ls[1], rs[1], neck[1]+10), 30)
    hip_ok = (lh[1] > y_top+60) and (rh[1] > y_top+60) and abs(lh[1]-rh[1]) < 120
    if hip_ok:
        y_bot = min(max(lh[1], rh[1]) + 45, 505)
        cx_bot = (lh[0]+rh[0])/2
        w_bot = max(abs(lh[0]-rh[0])*1.0, 80)
    else:
        y_bot = min(y_top + 290, 505)
        cx_bot = (ls[0]+rs[0])/2
        w_bot = max(abs(ls[0]-rs[0])*1.15, 90)
    if y_bot - y_top < 40: continue
    cx_top = (ls[0]+rs[0])/2
    w_top = max(abs(ls[0]-rs[0])*0.9, 70)
    def bc(t):
        cx = cx_top*(1-t)+cx_bot*t
        return np.array([cx, y_top*(1-t)+y_bot*t])
    def bh(t):
        return (w_top*(1-t)+w_bot*t)/2*1.55
    mask = Image.new("L",(W,H),0)
    dm = ImageDraw.Draw(mask)
    for r in range(90):
        t = r/89
        c = bc(t); hw = bh(t)*1.5
        dm.ellipse((c[0]-hw, c[1]-18, c[0]+hw, c[1]+18), fill=255)
    # 头保护
    hx, hy = (kk[0][0]+kk[1][0])/2, (kk[0][1]+kk[1][1])/2
    head_r = max(np.linalg.norm(neck-[hx,hy])*2.0, 100)
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
    wp = (np.asarray(mask) > 128).mean()
    if wp < 0.10 or wp > 0.45: continue  # 质量门
    base = f"dog_{n_ok:03d}"
    im512.save(os.path.join(OUT, base+"_src.png"))
    worn.save(os.path.join(OUT, base+"_worn.png"))
    mask.save(os.path.join(OUT, base+"_mask.png"))
    caps.write(f"{base}_worn.png\ta dog wearing a {cname} knitted sweater\n")
    n_ok += 1
caps.close()
print("LORA_DATA_DONE", n_ok, flush=True)
