# Style v5:官网图+品牌库弱标注训练
import json, os, glob
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 品牌库映射(品牌→风格/品类)
_raw = json.load(open(r"B:\mode\brand_library.json", encoding="utf-8"))
BLIB = [b for v in _raw.values() for b in v] if isinstance(_raw, dict) else _raw
STYLE2CAT = {"avant_garde":"dress","street":"top","minimal":"dress","elegant":"dress","casual":"top","vintage":"dress","sporty":"top","romantic":"dress","classic":"coat","bohemian":"dress"}
def brand_tags(brand_dir):
    name = brand_dir.replace("_", " ").lower()
    for b in BLIB:
        bn = str(b.get("品牌名", "")).lower()
        if bn and (bn in name or name in bn):
            st = str(b.get("风格标签", "")).lower()
            cats = [STYLE2CAT[st]] if st in STYLE2CAT else []
            price = str(b.get("均价档位", ""))
            cols = []
            if "轻奢" in price or "设计师" in price: cols = ["black","beige"]
            elif "奢" in price: cols = ["black","white"]
            else: cols = ["blue","white"]
            return {"cats": cats, "colors": cols}
    return None

CATS = ["dress","top","shirt","pants","skirt","coat","jacket","shorts","sweater"]
COLORS = ["black","white","gray","red","orange","yellow","green","blue","purple","pink","brown","beige"]
C2E = {"黑":"black","白":"white","灰":"gray","红":"red","橙":"orange","黄":"yellow","绿":"green","蓝":"blue","紫":"purple","粉":"pink","棕":"brown","米":"beige","卡其":"beige"}
CAT2E = {"上衣":"top","下装":"pants","半裙":"skirt","裤":"pants","裙":"dress","连衣裙":"dress","外套":"coat","大衣":"coat","T恤":"shirt"}

rows = []
for d in sorted(glob.glob(r"B:\mode\brand_site\*")):
    if not os.path.isdir(d): continue
    tags = brand_tags(os.path.basename(d))
    if not tags: continue
    cols = [C2E.get(c, c) for c in (tags.get("colors") or tags.get("主色") or []) if isinstance(c, str)]
    cats = [CAT2E.get(c, c) for c in (tags.get("cats") or tags.get("品类") or []) if isinstance(c, str)]
    # 也接受英文标签字段
    cols += [c for c in (tags.get("colors_en") or []) if c in COLORS]
    cats += [c for c in (tags.get("cats_en") or []) if c in CATS]
    for f in glob.glob(os.path.join(d, "img_*.*"))[:20]:
        rows.append((f, cats, cols))
print("官网弱标注:", len(rows), flush=True)

# 混合旧数据
v4 = json.load(open(r"B:\mode\style_train\data_v4\labels.json", encoding="utf-8"))
capmap = json.load(open(r"B:\mode\style_train\data_v4\caps.json", encoding="utf-8")) if os.path.exists(r"B:\mode\style_train\data_v4\caps.json") else {}
def cap_colors(txt):
    cmap = {"black":"black","white":"white","gray":"gray","grey":"gray","red":"red","orange":"orange","yellow":"yellow","green":"green","blue":"blue","purple":"purple","pink":"pink","brown":"brown","beige":"beige","cream":"beige","ivory":"beige"}
    return [v for k,v in cmap.items() if k in txt]
for d in v4:
    rows.append((os.path.join(r"B:\mode\style_train\data_v4", d["img"]), d["cats"], d.get("colors") or cap_colors(capmap.get(d["img"], ""))))

class DS(Dataset):
    def __init__(s, items):
        s.items = [r for r in items if os.path.exists(r[0])]
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, cats, cols = s.items[i % len(s.items)]
        try:
            img = Image.open(p).convert("RGB").resize((160,160))
        except Exception:
            img = Image.new("RGB", (160,160), (128,128,128))
        x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1)
        if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[2])
        yc = torch.tensor([1.0 if c in cats else 0.0 for c in CATS])
        yl = torch.tensor([1.0 if c in cols else 0.0 for c in COLORS])
        return x, yc, yl

class StyleNet(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.net = nn.Sequential(blk(3,48), blk(48,96), blk(96,192), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(192,128); s.hd_c = nn.Linear(128,9); s.hd_l = nn.Linear(128,12)
    def forward(s,x):
        f = torch.relu(s.fc(s.net(x).flatten(1)))
        return s.hd_c(f), s.hd_l(f)

tr = DS(rows)
va = DS(rows[::7])
m = StyleNet().cuda()
def pw(items, idx, names):
    cnt = {k:1 for k in names}
    for it in items:
        for k in it[idx]:
            if k in cnt: cnt[k]+=1
    tot = max(len(items),1)
    return torch.tensor([tot/cnt[k] for k in names]).cuda()
w_c = pw(rows, 1, CATS).clamp(1,15); w_l = pw(rows, 2, COLORS).clamp(1,15)
lf_c = nn.BCEWithLogitsLoss(pos_weight=w_c)
lf_l = nn.BCEWithLogitsLoss(pos_weight=w_l)
opt = torch.optim.AdamW(m.parameters(), lr=4e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=16)
best = 0
for ep in range(16):
    m.train()
    for x, yc, yl in DataLoader(tr, batch_size=48, shuffle=True):
        pc, pl = m(x.cuda())
        loss = lf_c(pc, yc.cuda()) + lf_l(pl, yl.cuda())
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    rc = {c:[0,0] for c in CATS}; rl = {c:[0,0] for c in COLORS}
    with torch.no_grad():
        for x, yc, yl in DataLoader(va, batch_size=64):
            pc, pl = [t.cpu() for t in m(x.cuda())]
            for j,c in enumerate(CATS):
                t = (yc[:,j]>0.5).sum().item()
                if t: rc[c][0]+=((torch.sigmoid(pc[:,j])>0.5)&(yc[:,j]>0.5)).sum().item(); rc[c][1]+=t
            for j,c in enumerate(COLORS):
                t = (yl[:,j]>0.5).sum().item()
                if t: rl[c][0]+=((torch.sigmoid(pl[:,j])>0.5)&(yl[:,j]>0.5)).sum().item(); rl[c][1]+=t
    mr = np.mean([v[0]/v[1] for v in rc.values() if v[1]>0]) if any(v[1]>0 for v in rc.values()) else 0
    lr_ = np.mean([v[0]/v[1] for v in rl.values() if v[1]>0]) if any(v[1]>0 for v in rl.values()) else 0
    print(f"ep{ep+1} catR={mr:.3f} colR={lr_:.3f}", flush=True)
    if mr+lr_ > best:
        best = mr+lr_
        torch.save(m.state_dict(), r"B:\mode\models\vyton_style_v5.pt")
    sch.step()
print("STYLE_V5_DONE", round(best,3), flush=True)
