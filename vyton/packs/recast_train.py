
# Style四头重蒸(从特征缓存,mbv3s基座)
import torch, torch.nn as nn
import numpy as np, os, json
X = np.load(r"B:\mode\style_train\feats_mbv3s.npy")
n = len(X)
# 标签重建(与提特征同序)
import glob
items = []
labs = json.load(open(r"B:\mode\style_train\pin_labels_v1.json", encoding="utf-8"))
for f, lab in labs.items():
    items.append((f, lab.get("cats", [])[:3], lab.get("colors", [])[:3], lab.get("style", [])[:2], lab.get("scene", [])[:2]))
for p in glob.glob(r"B:\mode\pinterest\board_imgs\*.jpg"):
    items.append((p, [], [], ["casual"], []))
yt = json.load(open(r"B:\mode\style_train\yt_labels.json", encoding="utf-8"))
bz = json.load(open(r"B:\mode\style_train\bz_labels.json", encoding="utf-8"))
for f, lab in {**yt, **bz}.items():
    st = lab.get("style", [])[:2]; sc = lab.get("scene", [])[:2]
    if st or sc:
        items.append((f, [], [], st, sc))
# 只留存在文件(提特征时同样过滤)——用数量对齐
CATS = ['dress','tops','outerwear','pants','skirts','shoes','bags','accessories','loungewear']
STYLES = ['casual','formal','street','vintage','minimalist','sporty','boho','elegant']
SCENES = ['work','casual','party','date','home','travel']
valid = [(i, it) for i, it in enumerate(items) if os.path.exists(it[0])][:n]
print(f"对齐: 特征{len(X)} 标签{len(valid)}", flush=True)
Ys = np.array([[1.0 if s in it[3] else 0. for s in STYLES] for _, it in valid], np.float32)
Ye = np.array([[1.0 if s in it[4] else 0. for s in SCENES] for _, it in valid], np.float32)
Xa = X[:len(valid)]
n_va = 250
Xt = torch.tensor(Xa[:-n_va]).cuda(); Xv = torch.tensor(Xa[-n_va:]).cuda()
Yst = torch.tensor(Ys[:-n_va]).cuda(); Ysv = torch.tensor(Ys[-n_va:]).cuda()
Yet = torch.tensor(Ye[:-n_va]).cuda(); Yev = torch.tensor(Ye[-n_va:]).cuda()
class Heads(nn.Module):
    def __init__(s, fin=Xa.shape[1]):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(fin, 384), nn.ReLU())
        s.hs = nn.Linear(384, 8); s.he = nn.Linear(384, 6)
    def forward(s, x):
        h = s.trunk(x)
        return s.hs(h), s.he(h)
net = Heads().cuda()
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
best = 0
for ep in range(400):
    net.train(); opt.zero_grad()
    ps, pe = net(Xt)
    loss = 0.8*nn.functional.binary_cross_entropy_with_logits(ps, Yst) + 0.8*nn.functional.binary_cross_entropy_with_logits(pe, Yet)
    loss.backward(); opt.step(); sched.step()
    if (ep+1) % 50 == 0:
        net.eval()
        with torch.no_grad():
            ps, pe = net(Xv)
            pr = (torch.sigmoid(ps) > 0.5).float()
            sr = ((pr==1)&(Ysv==1)).sum().item()/max((Ysv==1).sum().item(),1)
            pre = (torch.sigmoid(pe) > 0.5).float()
            er = ((pre==1)&(Yev==1)).sum().item()/max((Yev==1).sum().item(),1)
        print(f"ep{ep+1} styleR{sr:.3f} sceneR{er:.3f}", flush=True)
        if sr+er > best:
            best = sr+er
            torch.save(net.state_dict(), r"B:\mode\style_train\vyton_style_heads_mbv3s.pt")
print(f"RECAST_DONE best={best:.3f} headMB={os.path.getsize(r'G:/') if False else 0}", flush=True)
print("headMB:", round(os.path.getsize(r"B:\mode\style_train\vyton_style_heads_mbv3s.pt")/1e6, 2), flush=True)
