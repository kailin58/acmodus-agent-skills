
# v16盲测证据:随机8张验证图→v16四头预测→拼标注图
import sys, os, glob, random
sys.path.insert(0, r"B:\mode\vyton")
import torch, numpy as np
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

# 复刻v16的Net(从train脚本抄结构)
import importlib.util
spec = importlib.util.spec_from_file_location("t16", r"B:\mode\style_train\train_v16.py")
# 不exec整个训练!只抄Net定义:手动构建
from torchvision import models as tvm
CATS = ['dress','tops','outerwear','pants','skirts','shoes','bags','accessories','loungewear']
COLORS = ['black','white','gray','red','pink','blue','green','yellow','brown','beige','multi','denim']
STYLES = ['casual','formal','street','vintage','minimalist','sporty','boho','elegant']
SCENES = ['work','casual','party','date','home','travel']
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        m = tvm.efficientnet_b3(weights=None)
        f = m.classifier[1].in_features
        m.classifier = nn.Identity()
        s.back = m
        s.hd_cat = nn.Linear(f, len(CATS)); s.hd_col = nn.Linear(f, len(COLORS))
        s.hd_sty = nn.Linear(f, len(STYLES)); s.hd_sce = nn.Linear(f, len(SCENES))
    def forward(s, x):
        h = s.back.features(x)
        h = nn.functional.adaptive_avg_pool2d(h, 1).flatten(1)
        return s.hd_cat(h), s.hd_col(h), s.hd_sty(h), s.hd_sce(h)

net = Net()
sd = torch.load(r"B:\mode\models\vyton_style_v16_weighted.pt", map_location="cpu")
net.load_state_dict(sd, strict=False)
import sys as _s; _s.exit(0) if False else None
net.eval()
print("v16载入OK", flush=True)

# 验证图:从各源抽
random.seed(11)
pool = (glob.glob(r"B:\mode\pinterest\outfit_imgs\*.jpg")[:2000]
        + glob.glob(r"B:\mode\yt_outfit\*.jpg")[:500])
imgs = random.sample(pool, 8)
print("盲测8张:", flush=True)
rows = []
for p in imgs:
    x = torch.tensor(np.array(Image.open(p).convert("RGB").resize((160,160)))/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    with torch.no_grad():
        pc, pl, ps, pe = net(x)
    cat = CATS[pc.argmax()]
    col = COLORS[pl.argmax()]
    sty = STYLES[ps.argmax()]
    scn = SCENES[pe.argmax()]
    conf = torch.sigmoid(ps).max().item()
    print(f"  {os.path.basename(p)[:20]}: {cat}/{col}/{sty}/{scn} conf={conf:.2f}", flush=True)
    im = Image.open(p).convert("RGB").resize((220, 220))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 190, 220, 220], fill=(0,0,0))
    d.text((4, 194), f"{cat}|{col}|{sty}|{scn}", fill=(255,255,80))
    rows.append(im)
canvas = Image.new("RGB", (220*4, 220*2), (30,30,30))
for i, im in enumerate(rows):
    canvas.paste(im, ((i%4)*220, (i//4)*220))
canvas.save(r"B:\mode\style_train\v16_blind_evidence.png")
print("EVIDENCE_OK")
