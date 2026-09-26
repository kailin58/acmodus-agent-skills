
# Vyton-Style v4预热:品类头强化(官网数据到后可fine-tune)
import os, json, glob
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT = r"B:\mode\style_train"
import glob as _g
_BASE = r"B:\mode\vton_full\raw"
data = json.load(open(os.path.join(ROOT, "weak_labels.json"), encoding="utf-8"))
C2E = {"黑":"black","白":"white","灰":"gray","红":"red","橙":"orange","黄":"yellow","绿":"green","蓝":"blue","紫":"purple","粉":"pink","棕":"brown","米":"beige","卡其":"beige","裸":"beige"}
CAT2E = {"上衣":"top","下装":"bottom","半裙":"bottom","裤":"bottom","裙":"dress","连衣裙":"dress","外套":"outerwear","大衣":"outerwear","鞋":"shoes","包":"bag","配饰":"accessory","帽":"hat"}
for d in data:
    d["path"] = os.path.join(_BASE, d["img"])
    d["colors"] = [C2E.get(c, c) for c in d.get("colors", [])]
    d["cats"] = [CAT2E.get(c, c) for c in d.get("cats", [])]
print("数据:", len(data), flush=True)
CATS = ["top","bottom","dress","outerwear","shoes","bag","accessory","hat","other"]
COLORS = ["black","white","gray","red","orange","yellow","green","blue","purple","pink","brown","beige"]

class DS(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        d = self.rows[i % len(self.rows)]
        img = Image.open(d["path"]).convert("RGB").resize((160,160))
        x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1)
        import random
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            x = x * (0.85 + random.random()*0.3)
        yc = torch.tensor([1.0 if c in d.get("cats",[]) else 0.0 for c in CATS])
        ycol = torch.tensor([1.0 if c in d.get("colors",[]) else 0.0 for c in COLORS])
        return x.clamp(0,1), yc, ycol

class StyleNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(3,48,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48,96,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(96,192,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(192,128)
        s.head_cat = nn.Linear(128, 9)
        s.head_col = nn.Linear(128, 12)
    def forward(s,x):
        f = torch.relu(s.fc(s.net(x).flatten(1)))
        return s.head_cat(f), s.head_col(f)

n = len(data); sp = int(n*0.9)
tr = DS(data[:sp]); va = DS(data[sp:])
m = StyleNet().cuda()
# pos_weight双头(防塌缩,v2教训)
def pw(rows, keys, names):
    cnt = {k: 1 for k in names}
    for r in rows:
        for k in r.get(keys, []):
            if k in cnt: cnt[k] += 1
    tot = len(rows)
    return torch.tensor([tot/c for c in (cnt[k] for k in names)]).cuda()
w_cat = pw(data, "cats", CATS); w_col = pw(data, "colors", COLORS)
lf_cat = nn.BCEWithLogitsLoss(pos_weight=w_cat.clamp(1,12))
lf_col = nn.BCEWithLogitsLoss(pos_weight=w_col.clamp(1,12))
opt = torch.optim.AdamW(m.parameters(), lr=4e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
best = 0
for ep in range(10):
    m.train()
    for x, yc, ycol in DataLoader(tr, batch_size=48, shuffle=True):
        x, yc, ycol = x.cuda(), yc.cuda(), ycol.cuda()
        lc, lcol = m(x)
        loss = lf_cat(lc, yc) + lf_col(lcol, ycol)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    rec = {c: [0,0] for c in CATS}; recc = {c: [0,0] for c in COLORS}
    with torch.no_grad():
        for x, yc, ycol in DataLoader(va, batch_size=64):
            pc, pcol = [t.cpu() for t in m(x.cuda())]
            for j, c in enumerate(CATS):
                hit = ((torch.sigmoid(pc[:,j])>0.5) & (yc[:,j]>0.5)).sum().item()
                tot = (yc[:,j]>0.5).sum().item()
                if tot: rec[c][0]+=hit; rec[c][1]+=tot
            for j, c in enumerate(COLORS):
                hit = ((torch.sigmoid(pcol[:,j])>0.5) & (ycol[:,j]>0.5)).sum().item()
                tot = (ycol[:,j]>0.5).sum().item()
                if tot: recc[c][0]+=hit; recc[c][1]+=tot
    mr = np.mean([v[0]/v[1] for v in rec.values() if v[1]>0])
    cr = np.mean([v[0]/v[1] for v in recc.values() if v[1]>0])
    print(f"ep{ep+1} cat_R={mr:.3f} col_R={cr:.3f}", flush=True)
    if mr+cr > best:
        best = mr+cr
        torch.save(m.state_dict(), r"B:\mode\models\vyton_style_v4.pt")
sch.step()
print("STYLE_V4_DONE", round(best,3), flush=True)
