# Style v9:Pinterest意图真值+风格/场景多头
import glob, os, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models

CATS = ["dress","top","shirt","pants","skirt","coat","jacket","shorts","sweater"]
COLORS = ["black","white","gray","red","orange","yellow","green","blue","purple","pink","brown","beige"]
STYLES = ["minimalist","french","vintage","street","sporty","elegant","casual","boho"]  # 8风格(收敛)
SCENES = ["work","date","party","casual","travel","home"]  # 6场景

# 风格词→8类归一
S_MAP = {"minimalist":"minimalist","scandinavian":"minimalist","normcore":"minimalist","极简":"minimalist",
         "french":"elegant","italian":"elegant","法式":"elegant",
         "vintage":"vintage","american vintage":"vintage","y2k":"vintage","复古":"vintage","美式":"vintage",
         "streetwear":"street","street":"street","techwear":"street","japanese":"street","korean":"street","日系":"street","韩系":"street",
         "athleisure":"sporty","sporty":"sporty","gorpcore":"sporty","运动":"sporty",
         "boho":"boho","cottagecore":"boho","pastel":"boho",
         "smart casual":"casual","business casual":"casual","casual":"casual","休闲":"casual",
         "goth":"street","preppy":"elegant","cityboy":"street"}
C_MAP = {"work":"work","office":"work","interview":"work","通勤":"work","职场":"work","business":"work",
         "date":"date","dinner":"date","brunch":"date","约会":"date",
         "party":"party","concert":"party","wedding":"party",
         "casual":"casual","weekend":"casual","school":"casual","休闲":"casual","学生":"casual",
         "travel":"travel","holiday":"travel","度假":"travel",
         "home":"home","stayhome":"home"}

labs = json.load(open(r"B:\mode\style_train\pin_labels_v1.json", encoding="utf-8"))
clean = set(json.load(open(r"B:\mode\style_train\pin_clean_v1.json", encoding="utf-8")))
items = []
for f, lab in labs.items():
    if f not in clean: continue
    st = list({S_MAP[s] for s in lab["style"] if s in S_MAP})[:2]
    sc = list({C_MAP[c] for c in lab["scene"] if c in C_MAP})[:2]
    if not st and not sc:
        continue
    col = []
    items.append((f, [], col, st, sc))
print("Pinterest条目:", len(items), flush=True)

# 合并旧数据(VITON教师971+民族421)只供cat/color(无风格标注)
for p in sorted(glob.glob(r"B:\mode\vton_training\teacher_pairs_v2\t_0*.png"))[:400]:
    items.append((p, ["dress"], [], [], []))
ed = sorted(glob.glob(r"B:\mode\ethnic_data\*\*.jpg"))
for p in ed[:200]:
    items.append((p, ["dress"], [], [], []))

# 画板数据(板名=品类真值,最干净标签)
board_items = json.load(open(r"B:\mode\style_train\board_items.json", encoding="utf-8"))
for p, cats_b in board_items:
    items.append((p, cats_b, [], [], []))
print("画板条目:", len(board_items), flush=True)

# v13自评环修正:剔除低信息+高置信冲突样本标签采纳模型预测
import json as _j2
sus = _j2.load(open(r"B:\mode\style_train\suspect_v12.json", encoding="utf-8"))
lowinfo = {x["f"] for x in sus if x["type"] == "lowinfo"}
fix = {x["f"]: x for x in sus if x["type"] == "style" and x["conf"] > 0.6}
items = [it for it in items if it[0] not in lowinfo]
for i, (f, cats, cols, st, sc) in enumerate(items):
    if f in fix:
        items[i] = (f, cats, cols, [fix[f]["pred"]], sc)
print(f"v13修正: 剔低信息{len(lowinfo)} 改标签{len(fix)}", flush=True)
print("总条目:", len(items), flush=True)

class DS(Dataset):
    def __init__(s, its):
        s.its = [r for r in its if os.path.exists(r[0])]
    def __len__(s): return len(s.its)
    def __getitem__(s, i):
        p, cats, cols, st, sc = s.its[i % len(s.its)]
        try:
            img = Image.open(p).convert("RGB").resize((160,160))
        except Exception:
            img = Image.new("RGB", (160,160), (128,128,128))
        x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1)
        if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[2])
        yc = torch.tensor([1.0 if c in cats else 0.0 for c in CATS])
        yl = torch.tensor([1.0 if c in cols else 0.0 for c in COLORS])
        ys = torch.tensor([1.0 if c in st else 0.0 for c in STYLES])
        ye = torch.tensor([1.0 if c in sc else 0.0 for c in SCENES])
        return x, yc, yl, ys, ye

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        m = models.efficientnet_b3(weights=None)
        try:
            from torchvision.models import EfficientNet_B3_Weights
            m = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        except Exception:
            pass
        s.back = m
        f = m.classifier[1].in_features
        s.hd_cat = nn.Linear(f, len(CATS))
        s.hd_col = nn.Linear(f, len(COLORS))
        s.hd_sty = nn.Linear(f, len(STYLES))
        s.hd_sce = nn.Linear(f, len(SCENES))
    def forward(s, x):
        h = s.back.features(x)
        h = nn.functional.adaptive_avg_pool2d(h, 1).flatten(1)
        return s.hd_cat(h), s.hd_col(h), s.hd_sty(h), s.hd_sce(h)

va_split = max(1, int(len(items)*0.12))
va_items = items[:va_split]
tr_items = items[va_split:]
tr = DataLoader(DS(tr_items), batch_size=32, shuffle=True)
va = DataLoader(DS(va_items), batch_size=32)

net = Net().cuda()
opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
lf = nn.BCEWithLogitsLoss()
best = 0
import torch.nn.functional as F
for ep in range(12):
    net.train()
    for x, yc, yl, ys, ye in tr:
        x = x.cuda()
        pc, pl, ps, pe = net(x)
        loss = 0.4*lf(pc, yc.cuda()) + 0.4*lf(pl, yl.cuda()) + 0.8*lf(ps, ys.cuda()) + 0.8*lf(pe, ye.cuda())
        opt.zero_grad(); loss.backward(); opt.step()
    sch.step()
    net.eval()
    with torch.no_grad():
        rs, re_, rc, rl = {k:[0,0] for k in STYLES}, {k:[0,0] for k in SCENES}, {k:[0,0] for k in CATS}, {k:[0,0] for k in COLORS}
        for x, yc, yl, ys, ye in va:
            pc, pl, ps, pe = net(x.cuda())
            for j, k in enumerate(STYLES):
                t = (ys[:,j]>0.5).sum().item()
                if t: rs[k][0] += ((torch.sigmoid(ps[:,j])>0.5).cpu()&(ys[:,j]>0.5)).sum().item(); rs[k][1] += t
            for j, k in enumerate(SCENES):
                t = (ye[:,j]>0.5).sum().item()
                if t: re_[k][0] += ((torch.sigmoid(pe[:,j])>0.5).cpu()&(ye[:,j]>0.5)).sum().item(); re_[k][1] += t
        sr = np.mean([v[0]/v[1] for v in rs.values() if v[1]>0]) if any(v[1]>0 for v in rs.values()) else 0
        er = np.mean([v[0]/v[1] for v in re_.values() if v[1]>0]) if any(v[1]>0 for v in re_.values()) else 0
        print(f"ep{ep+1} styleR={sr:.3f} sceneR={er:.3f}", flush=True)
        if sr+er > best:
            best = sr+er
            torch.save(net.state_dict(), r"B:\mode\models\vyton_style_v13_multi.pt")
            print(f"  BEST {sr+er:.3f}", flush=True)

try:
    import sys
    sys.path.insert(0, r"B:\mode\rsi")
    from rsi_hook import rsi_log
    rsi_log("style", "v13_自评环", f"联合{best:.3f}", "v12自评74可疑+修正", "自评环首跑:剔12低信息+高置信改标")
except Exception:
    pass

print("STYLE_V9_DONE", flush=True)
