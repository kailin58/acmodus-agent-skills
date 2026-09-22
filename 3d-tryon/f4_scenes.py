
# F4场景引擎:三场景(白棚/街拍/舞台)+光照+地面
import os, math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import body_engine as anny

OUT = r"B:\mode\vton_training\garment_poc"
SCENES = os.path.join(OUT, "scenes")
os.makedirs(SCENES, exist_ok=True)

model = body_engine.Engine(local_changes="default")
NB = model.bone_count
res0 = model(None, {"height": 0.5, "weight": 0.5})
V0 = np.asarray(res0["vertices"])[0]
bh = np.asarray(res0["rest_bone_heads"])[0]

# 红裙纹理参数
z_min, z_max = V0[:,2].min(), V0[:,2].max()
x_min, x_max = V0[:,0].min(), V0[:,0].max()
h = z_max - z_min
shoulder_z, hem_z = z_max - h*0.15, z_max - h*0.62
G = Image.open(os.path.join(OUT, "dress_red.png")).convert("RGB").resize((200,260))
garr = np.array(G)
gw, gh = 200, 260
def colorize(V):
    colors = np.full((len(V), 3), 90, np.uint8)
    for idx in range(len(V)):
        z = V[idx,2]
        if not (hem_z < z < shoulder_z): continue
        layer = V[np.abs(V[:,2]-z) < h*0.012]
        if len(layer) < 3: continue
        lx0, lx1 = layer[:,0].min(), layer[:,0].max()
        u = int(np.clip((V[idx,0]-lx0)/(lx1-lx0+1e-9)*(gw-1),0,gw-1))
        v = int(np.clip((shoulder_z-z)/(shoulder_z-hem_z)*(gh-1),0,gh-1))
        r_,g_,b_ = garr[v,u]
        if r_>225 and g_>225 and b_>225: continue
        colors[idx] = (r_,g_,b_)
    return colors

colors = colorize(V0)

def render(scene):
    W, H = 480, 720
    if scene == "studio":
        bg = Image.new("RGB", (W,H), (238,238,240))
        d = ImageDraw.Draw(bg)
        for i in range(H):
            t = i/H
            d.line([(0,i),(W,i)], fill=(int(238-20*t),)*3)
        floor_y = int(H*0.82)
        d.rectangle([0,floor_y,W,H], fill=(210,210,214))
        # 地面阴影
        sh = Image.new("RGB",(W,H),(0,0,0))
        ds = ImageDraw.Draw(sh)
        cx = W//2
        ds.ellipse([cx-70,floor_y+8,cx+70,floor_y+38], fill=(60,60,64))
        sh = sh.filter(ImageFilter.GaussianBlur(14))
        bg = Image.blend(bg, sh.point(lambda p:min(255,int(p*1.8))), 0.25)
    elif scene == "street":
        bg = Image.new("RGB", (W,H), (168,178,192))
        d = ImageDraw.Draw(bg)
        for i in range(0,H,36):
            d.line([(0,i),(W,i)], fill=(150,160,175))
        d.rectangle([0,int(H*0.55),W,int(H*0.62)], fill=(120,130,148))
        for bx in range(30, W, 90):
            d.rectangle([bx,int(H*0.30),bx+54,int(H*0.62)], fill=(126,136,152))
            d.rectangle([bx+8,int(H*0.34),bx+24,int(H*0.46)], fill=(198,212,228))
            d.rectangle([bx+30,int(H*0.34),bx+46,int(H*0.46)], fill=(198,212,228))
        d.rectangle([0,int(H*0.62),W,H], fill=(98,100,106))
        for ry in range(int(H*0.62)+8, H, 30):
            d.line([(0,ry),(W,ry)], fill=(88,90,96))
    else:  # stage
        from PIL import ImageDraw as ID
        bg = Image.new("RGB", (W,H), (16,12,24))
        d = ID.Draw(bg)
        for y in range(H):
            t = y/H
            d.line([(0,y),(W,y)], fill=(int(16+40*t), int(12+20*t), int(24+50*t)))
        # 聚光灯锥
        spot = Image.new("RGB",(W,H),(0,0,0))
        dsp = ID.Draw(spot)
        cx = W//2
        dsp.polygon([(cx-30,0),(cx+30,0),(cx+190,H),(cx-190,H)], fill=(90,70,30))
        spot = spot.filter(ImageFilter.GaussianBlur(30))
        bg = Image.blend(bg, spot.point(lambda p:p), 0.5)
        d = ID.Draw(bg)
        d.ellipse([cx-140,int(H*0.86),cx+140,int(H*0.90)], outline=(255,220,150), width=3)
    # 人物点云
    d = ImageDraw.Draw(bg)
    ang = math.radians(10)
    xr = V0[:,0]*math.cos(ang) - V0[:,1]*math.sin(ang)
    depth = V0[:,0]*math.sin(ang) + V0[:,1]*math.cos(ang)
    rx0, rx1 = xr.min(), xr.max()
    px = ((xr-rx0)/(rx1-rx0)*(W-160)+80).astype(int)
    py = ((z_max-V0[:,2])/h*(H-140)+70).astype(int)
    for i in np.argsort(depth):
        base = tuple(int(c) for c in colors[i])
        # 场景光照调制
        if scene=="street": base=tuple(min(255,int(c*1.05)) for c in base)
        if scene=="stage": base=tuple(min(255,int(c*0.85+b*0.15)) for c,b in zip(base,(255,200,120)))
        d.ellipse((px[i]-1,py[i]-1,px[i]+1,py[i]+1), fill=base)
    return bg

for s in ["studio","street","stage"]:
    img = render(s)
    img.save(os.path.join(SCENES, f"scene_{s}.png"))
    print("SCENE", s, flush=True)
print("F4_DONE", flush=True)
