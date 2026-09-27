# Style v4正式:929 VITON品类 + 150旧素材(颜色)混合
import os, json, glob
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT = r"B:\mode\style_train"
v4 = json.load(open(os.path.join(ROOT, "data_v4", "labels.json"), encoding="utf-8"))
old = json.load(open(os.path.join(ROOT, "weak_labels.json"), encoding="utf-8"))
CATS = ["dress","top","shirt","pants","skirt","coat","jacket","shorts","sweater"]
COLORS = ["black","white","gray","red","orange","yellow","green","blue","purple","pink","brown","beige"]
C2E = {"黑":"black","白":"white","灰":"gray","红":"red","橙":"orange","黄":"yellow","绿":"green","蓝":"blue","紫":"purple","粉":"pink","棕":"brown","米":"beige","卡其":"beige"}
CAT2E = {"上衣":"top","下装":"pants","半裙":"skirt","裤":"pants","裙":"dress","连衣裙":"dress","外套":"coat","大衣":"coat","鞋":"dress","包":"top","配饰":"top","帽":"top","T恤":"shirt"}
rows = []
for d in v4:
    rows.append((os.path.join(ROOT, "data_v4", d["img"]), d["cats"], d.get("colors", [])))
base_old = r"B:\mode\vton_full\raw"
for d in old:
    cs = [C2E.get(c, c) for c in d.get("colors", [])]
    ct = [CAT2E.get(c, c) for c in d.get("cats", [])]
    rows.append((os.path.join(base_old, d["img"]), ct, cs))
print("混合数据:", len(rows), flush=True)

class DS(Dataset):
    def __init__(s, items):
        s.items = [r for r in items if os.path.exists(r[0])]
        print("有效:", len(s.items), flush=True)
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, cats, cols = s.items[i % len(s.items)]
        img = Image.open(p).convert("RGB").resize((160,160))
        x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1)
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2]) * (0.85+torch.rand(1).item()*0.3)
        yc = torch.tensor([1.0 if c in cats else 0.0 for c in CATS])
        ycol = torch.tensor([1.0 if c in cols else 0.0 for c in COLORS])
        return x.clamp(0,1), yc, ycol

class StyleNet(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.net = nn.Sequential(blk(3,48), blk(48,96), blk(96,192), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(192,128); s.hd_c = nn.Linear(128,9); s.hd_l = nn.Linear(128,12)
    def forward(s,x):
        f = torch.relu(s.fc(s.net(x).flatten(1)))
        return s.hd_c(f), s.hd_l(f)

allrows = DS(rows)
n = len(allrows); sp = int(n*0.92)
tr = allrows; va = DS(rows[sp:])
m = StyleNet().cuda()
def pw(items, keys, names):
    cnt = {k:1 for k in names}
    for _, ks, _2 in items:
        for k in ks:
            if k in cnt: cnt[k]+=1
    tot = max(len(items),1)
    return torch.tensor([tot/cnt[k] for k in names]).cuda()
w_c = pw(rows, 1, CATS); w_l = pw(rows, 2, COLORS)
lf_c = nn.BCEWithLogitsLoss(pos_weight=w_c.clamp(1,15))
lf_l = nn.BCEWithLogitsLoss(pos_weight=w_l.clamp(1,15))
opt = torch.optim.AdamW(m.parameters(), lr=4e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=14)
best = 0
for ep in range(14):
    m.train()
    for x, yc, yl in DataLoader(tr, batch_size=48, shuffle=True):
        x, yc, yl = x.cuda(), yc.cuda(), yl.cuda()
        pc, pl = m(x)
        loss = lf_c(pc, yc) + lf_l(pl, yl)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    rc = {c:[0,0] for c in CATS}; rl = {c:[0,0] for c in COLORS}
    with torch.no_grad():
        for x, yc, yl in DataLoader(va, batch_size=64):
            pc, pl = [t.cpu() for t in m(x.cuda())]
            for j,c in enumerate(CATS):
                t = (yc[:,j]>0.5).sum().item()
                if t: rc[c][0] += (((torch.sigmoid(pc[:,j])>0.5)&(yc[:,j]>0.5)).sum().item()); rc[c][1]+=t
            for j,c in enumerate(COLORS):
                t = (yl[:,j]>0.5).sum().item()
                if t: rl[c][0] += (((torch.sigmoid(pl[:,j])>0.5)&(yl[:,j]>0.5)).sum().item()); rl[c][1]+=t
    mr = np.mean([v[0]/v[1] for v in rc.values() if v[1]>0]) if any(v[1]>0 for v in rc.values()) else 0
    lr_ = np.mean([v[0]/v[1] for v in rl.values() if v[1]>0]) if any(v[1]>0 for v in rl.values()) else 0
    print(f"ep{ep+1} catR={mr:.3f} colR={lr_:.3f}", flush=True)
    if mr+lr_ > best:
        best = mr+lr_
        torch.save(m.state_dict(), r"B:\mode\models\vyton_style_v4f.pt")
    sch.step()
print("STYLE_V4F_DONE", round(best,3), flush=True)
