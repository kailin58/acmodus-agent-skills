# v19修:旧特征标签按v18_gap原valid序重建(当时items序:pin→board(当时全部)→yt→bz)
import torch, torch.nn as nn
import numpy as np, os, json, glob as g

X1 = np.load(r"B:/mode/style_train/feats_enb3_gap.npy")
X2 = np.load(r"B:/mode/style_train/board_feats.npy")
names2 = json.load(open(r"B:/mode/style_train/board_names.json"))
labs = json.load(open(r"B:/mode/style_train/pin_labels_v1.json", encoding="utf-8"))
yt = json.load(open(r"B:/mode/style_train/yt_labels.json", encoding="utf-8"))
bz = json.load(open(r"B:/mode/style_train/bz_labels.json", encoding="utf-8"))

# v18_gap当时items序(脚本原样):pin→board glob→yt/bz合并;截到len(X1)
items = []
for f, lab in labs.items():
    items.append((f, lab.get("style", [])[:2], lab.get("scene", [])[:2]))
board_all = sorted(g.glob(r"B:/mode/pinterest/board_imgs/*.jpg"))
for p in board_all:
    items.append((p, ["casual"], []))
for f, lab in {**yt, **bz}.items():
    st = lab.get("style", [])[:2]; sc = lab.get("scene", [])[:2]
    if st or sc:
        items.append((f, st, sc))
valid1 = [(f, st, sc) for f, st, sc in items if os.path.exists(f)][:len(X1)]
print("旧对齐", len(valid1), "/", X1.shape[0])
if len(valid1) != X1.shape[0]:
    print("MISMATCH! 退出防错训")
    raise SystemExit

STYLES = ['casual','formal','street','vintage','minimalist','sporty','boho','elegant']
SCENES = ['work','casual','party','date','home','travel']
X = np.concatenate([X1, X2], 0)
Ys, Ye, W = [], [], []
for f, st, sc in valid1:
    Ys.append([1.0 if s in st else 0. for s in STYLES])
    Ye.append([1.0 if s in sc else 0. for s in SCENES])
    W.append(1.0 if f.startswith("B:/mode/pinterest/outfit") or "outfit" in f else 0.5)
for n in names2:
    Ys.append([1.0,0,0,0,0,0,0,0]); Ye.append([0]*6); W.append(0.3)
Ys = np.array(Ys, np.float32); Ye = np.array(Ye, np.float32); W = np.array(W, np.float32)
print("总", X.shape)

# 验证集从有真标部分(前段)切,弱标板图全入训练
n_va = 400
Xv = torch.tensor(X[:n_va]).cuda()
Ysv = torch.tensor(Ys[:n_va]).cuda(); Yev = torch.tensor(Ye[:n_va]).cuda()
Xt = torch.tensor(X[n_va:]).cuda()
Yst = torch.tensor(Ys[n_va:]).cuda(); Yet = torch.tensor(Ye[n_va:]).cuda()
Wt = torch.tensor(W[n_va:]).cuda()
class H(nn.Module):
    def __init__(s, fin=1536):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(fin, 384), nn.ReLU())
        s.hs = nn.Linear(384, 8); s.he = nn.Linear(384, 6)
    def forward(s, x):
        h = s.trunk(x); return s.hs(h), s.he(h)
net = H(1536).cuda()
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
best = 0
for ep in range(400):
    net.train(); opt.zero_grad()
    ps, pe = net(Xt)
    ls = (nn.functional.binary_cross_entropy_with_logits(ps, Yst, reduction="none").mean(1) * Wt).mean()
    le = (nn.functional.binary_cross_entropy_with_logits(pe, Yet, reduction="none").mean(1) * Wt).mean()
    loss = 0.8*ls + 0.8*le
    loss.backward(); opt.step(); sched.step()
    if (ep+1) % 100 == 0:
        net.eval()
        with torch.no_grad():
            ps, pe = net(Xv)
            sr = (((torch.sigmoid(ps)>0.5).float()==1)&(Ysv==1)).sum().item()/max((Ysv==1).sum().item(),1)
            er = (((torch.sigmoid(pe)>0.5).float()==1)&(Yev==1)).sum().item()/max((Yev==1).sum().item(),1)
        print(f"ep{ep+1} styleR{sr:.3f} sceneR{er:.3f}", flush=True)
        if sr+er > best:
            best = sr+er
            torch.save(net.state_dict(), r"B:/mode/style_train/vyton_style_v19_heads.pt")
print(f"V19_DONE best={best:.3f} (v18gap=1.120)")
