# TabooNet v1:全禁忌单模型检测(12维)
import glob, os, json, random, math
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont

CLASSES = ["funeral","religious_sacred","political","flag_misuse","exposure",
           "gesture_offense","animal_omen","number_taboo","food_violation",
           "material_risk","gender_mismatch","SAFE"]

rng = random.Random(42)
W = H = 128
def base_canvas():
    return Image.new("RGB", (W,H), (rng.randint(180,255),)*3)

def draw_funeral():
    im = base_canvas(); d = ImageDraw.Draw(im)
    cx, cy = 64, 64
    for ang in range(0, 360, 20):
        x2 = cx + 40*math.cos(math.radians(ang)); y2 = cy + 40*math.sin(math.radians(ang))
        d.line((cx, cy, x2, y2), fill=(245,245,250), width=7)
    d.ellipse((50,50,78,78), fill=(230,200,50))
    return im

def draw_religious():
    im = base_canvas(); d = ImageDraw.Draw(im)
    d.polygon([(64,20),(40,70),(88,70)], fill=(255,200,220))
    d.ellipse((44,60,84,100), fill=(240,180,210))
    d.rectangle((20,100,108,128), fill=(255,220,80))
    return im

def draw_political():
    im = Image.new("RGB",(W,H),(200,30,30)); d = ImageDraw.Draw(im)
    d.polygon([(30,30),(98,30),(30,98)], fill=(20,20,20))
    d.polygon([(98,98),(98,40),(40,98)], fill=(20,20,20))
    return im

def draw_flag():
    im = Image.new("RGB",(W,H),(20,120,40)); d = ImageDraw.Draw(im)
    d.rectangle((10,50,118,66), fill=(245,245,245))
    d.rectangle((10,72,118,76), fill=(245,245,245))
    return im

def draw_exposure():
    im = Image.new("RGB",(W,H),(240,200,170)); d = ImageDraw.Draw(im)
    d.rectangle((30,40,98,128), fill=(235,190,160))
    return im

def draw_gesture():
    im = base_canvas(); d = ImageDraw.Draw(im)
    d.ellipse((40,40,88,88), outline=(120,90,70), width=10)
    d.line((64,88,64,120), fill=(120,90,70), width=12)
    return im

def draw_owl():
    im = Image.new("RGB",(W,H),(40,40,60)); d = ImageDraw.Draw(im)
    d.ellipse((34,30,94,110), fill=(70,60,80))
    d.ellipse((44,50,62,68), fill=(240,230,140)); d.ellipse((66,50,84,68), fill=(240,230,140))
    d.ellipse((51,56,56,62), fill=(20,20,20)); d.ellipse((73,56,78,62), fill=(20,20,20))
    return im

def draw_number():
    im = base_canvas(); d = ImageDraw.Draw(im)
    try: font = ImageFont.truetype("arial.ttf", 80)
    except: font = ImageFont.load_default()
    d.text((44,24), "4", fill=(220,30,30), font=font)
    return im

def draw_food():
    im = base_canvas(); d = ImageDraw.Draw(im)
    d.ellipse((20,40,70,90), fill=(250,170,170))
    d.polygon([(80,40),(80,100),(110,100)], fill=(150,60,60))
    return im

def draw_leather():
    im = Image.new("RGB",(W,H),(160,110,70)); d = ImageDraw.Draw(im)
    for _ in range(14):
        x, y = rng.randint(10,118), rng.randint(10,118)
        d.arc((x,y,x+14,y+14), 0, 180, fill=(130,85,50), width=3)
    return im

def draw_gender():
    im = base_canvas(); d = ImageDraw.Draw(im)
    d.ellipse((30,30,54,54), outline=(60,90,200), width=5)
    d.line((50,50,68,68), fill=(60,90,200), width=5)
    d.polygon([(78,40),(64,110),(96,110)], fill=(230,120,160))
    return im

GENS = [draw_funeral, draw_religious, draw_political, draw_flag,
        draw_exposure, draw_gesture, draw_owl, draw_number,
        draw_food, draw_leather, draw_gender]

def make_synthetic(n_per=60):
    items = []
    for ci, g in enumerate(GENS):
        for _ in range(n_per):
            items.append((g(), ci))
    return items

syn = make_synthetic()
print("合成正例:", len(syn), flush=True)

neg_paths = []
for d in glob.glob(r"B:\mode\ethnic_data\*"):
    neg_paths += glob.glob(os.path.join(d, "*.jpg"))[:4]
neg_paths += glob.glob(r"B:\mode\vton_training\teacher_pairs_v2\t_01*.png")[:40]
print("负例:", len(neg_paths), flush=True)

class TDS(Dataset):
    def __len__(s): return len(s.pairs)
    def __init__(s, pairs): s.pairs = pairs
    def __getitem__(s, i):
        img_or_path, y = s.pairs[i]
        if isinstance(img_or_path, str):
            img = Image.open(img_or_path).convert("RGB").resize((128,128))
        else:
            img = img_or_path
        x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1)
        if torch.rand(1).item()<0.5: x = torch.flip(x, dims=[2])
        yv = torch.zeros(12); yv[11 if y==-1 else y] = 1.0
        return x, yv

pairs = [(im, c) for im, c in syn] + [(p, -1) for p in neg_paths]
random.shuffle(pairs)
cut = len(pairs)*9//10
tr = TDS(pairs[:cut]); va = TDS(pairs[cut:])

class TabooNet(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,2,1), nn.BatchNorm2d(o), nn.ReLU())
        s.net = nn.Sequential(blk(3,32), blk(32,64), blk(64,128), blk(128,192), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(192, 12)
    def forward(s, x): return s.fc(s.net(x).flatten(1))

m = TabooNet().cuda()
opt = torch.optim.AdamW(m.parameters(), lr=6e-4)
lf = nn.BCEWithLogitsLoss()
best = 0
for ep in range(10):
    m.train()
    for x, y in DataLoader(tr, batch_size=48, shuffle=True):
        loss = lf(m(x.cuda()), y.cuda())
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); c=t=0
    with torch.no_grad():
        for x, y in DataLoader(va, batch_size=64):
            p = m(x.cuda()).argmax(1).cpu()
            c += (p == y.argmax(1)).sum().item(); t += len(y)
    acc = c/t
    print(f"ep{ep+1} acc={acc:.3f}", flush=True)
    if acc > best:
        best = acc
        torch.save(m.state_dict(), r"B:\mode\models\vyton_taboo_net.pt")
print("TABOO_DONE", round(best,3), flush=True)