
# 尺寸驱动体型验证:三体型并排渲染
import os, math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import body_engine as anny

model = anny.Anny(local_changes="default")
bodies = [
    ({"height": 0.9, "weight": 0.15, "muscle": 0.3}, "slim-tall"),
    ({"height": 0.5, "weight": 0.5, "muscle": 0.5}, "standard"),
    ({"height": 0.4, "weight": 0.85, "muscle": 0.85}, "strong"),
]
frames = []
for kw, name in bodies:
    res = model(None, kw)
    V = np.asarray(res["vertices"])[0]
    W, H = 300, 620
    img = Image.new("RGB", (W,H), (24,24,28))
    d = ImageDraw.Draw(img)
    x0, x1 = V[:,0].min(), V[:,0].max()
    z0, z1 = V[:,2].min(), V[:,2].max()
    px = ((V[:,0]-x0)/(x1-x0+1e-9)*(W-60)+30).astype(int)
    py = ((z1-V[:,2])/(z1-z0+1e-9)*(H-100)+60).astype(int)
    order = np.argsort(V[:,1])
    dn = (V[:,1][order]-V[:,1].min())/(V[:,1].ptp() if hasattr(V[:,1],'ptp') else np.ptp(V[:,1])+1e-9)
    for j, i in enumerate(order):
        b = int(90+dn[j]*140)
        d.ellipse((px[i]-1,py[i]-1,px[i]+1,py[i]+1), fill=(int(b*0.75),int(b*0.87),min(255,b+30)))
    try: font = ImageFont.truetype("arial.ttf", 16)
    except: font = ImageFont.load_default()
    d.text((20,20), name, fill=(240,240,240), font=font)
    frames.append(img)

combo = Image.new("RGB", (300*3, 620), (14,14,18))
for i, im in enumerate(frames):
    combo.paste(im, (i*300, 0))
combo.save(r"B:\mode\vton_training\garment_poc\body_types.png")
print("BODY_TYPES_DONE", flush=True)
