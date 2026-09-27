# Style v4f快速评估
import os, json
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
CAT2E = {"上衣":"top","下装":"pants","半裙":"skirt","裤":"pants","裙":"dress","连衣裙":"dress","外套":"coat","大衣":"coat","T恤":"shirt"}
rows = []
for d in v4: rows.append((os.path.join(ROOT, "data_v4", d["img"]), d["cats"], d.get("colors", [])))
base_old = r"B:\mode\vton_full\raw"
for d in old:
    rows.append((os.path.join(base_old, d["img"]), [CAT2E.get(c,c) for c in d.get("cats",[])], [C2E.get(c,c) for c in d.get("colors",[])]))
rows = [r for r in rows if os.path.exists(r[0])]
class VA(Dataset):
    def __init__(s, items): s.items = items
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, cats, cols = s.items[i]
        img = Image.open(p).convert("RGB").resize((160,160))
        x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1)
        return x, torch.tensor([1.0 if c in cats else 0.0 for c in CATS]), torch.tensor([1.0 if c in cols else 0.0 for c in COLORS])
class StyleNet(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.net = nn.Sequential(blk(3,48), blk(48,96), blk(96,192), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(192,128); s.hd_c = nn.Linear(128,9); s.hd_l = nn.Linear(128,12)
    def forward(s,x):
        f = torch.relu(s.fc(s.net(x).flatten(1)))
        return s.hd_c(f), s.hd_l(f)
m = StyleNet().cuda()
m.load_state_dict(torch.load(r"B:\mode\models\vyton_style_v4f.pt", map_location="cuda", weights_only=True))
m.eval()
va = VA(rows[::7])
rc = {c:[0,0] for c in CATS}; rl = {c:[0,0] for c in COLORS}
with torch.no_grad():
    for x, yc, yl in DataLoader(va, batch_size=64):
        pc, pl = [t.cpu() for t in m(x.cuda())]
        for j,c in enumerate(CATS):
            t=(yc[:,j]>0.5).sum().item()
            if t: rc[c][0]+=((torch.sigmoid(pc[:,j])>0.5)&(yc[:,j]>0.5)).sum().item(); rc[c][1]+=t
        for j,c in enumerate(COLORS):
            t=(yl[:,j]>0.5).sum().item()
            if t: rl[c][0]+=((torch.sigmoid(pl[:,j])>0.5)&(yl[:,j]>0.5)).sum().item(); rl[c][1]+=t
mr = np.mean([v[0]/v[1] for v in rc.values() if v[1]>0])
hit = sum(v[1] for v in rc.values())
print("品类召回:", round(mr,3), "| 有品类标签样本命中数:", hit)
avail = [f"{c}:{v[0]}/{v[1]}" for c,v in rc.items() if v[1]>0]
print(" | ".join(avail[:9]))
