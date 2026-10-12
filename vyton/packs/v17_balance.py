# Style v17:负载均衡损失实验(MoE思想#4)
# 对照:v16口径(enb3特征+四头,无均衡) vs v17(同+梯度均衡权重)
import torch, torch.nn as nn
import numpy as np, os, json, glob
from PIL import Image
from torchvision import models as tvm, transforms

# enb3特征提取(与fair_cmp同法)
labs = json.load(open(r"B:\mode\style_train\pin_labels_v1.json", encoding="utf-8"))
items = []
for f, lab in labs.items():
    items.append((f, lab.get("style", [])[:2], lab.get("scene", [])[:2]))
for p in glob.glob(r"B:\mode\pinterest\board_imgs\*.jpg"):
    items.append((p, ["casual"], []))
yt = json.load(open(r"B:\mode\style_train\yt_labels.json", encoding="utf-8"))
bz = json.load(open(r"B:\mode\style_train\bz_labels.json", encoding="utf-8"))
for f, lab in {**yt, **bz}.items():
    st = lab.get("style", [])[:2]; sc = lab.get("scene", [])[:2]
    if st or sc:
        items.append((f, st, sc))
valid = [it for it in items if os.path.exists(it[0])]
print(f"items {len(valid)}", flush=True)
m = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.IMAGENET1K_V1)
m.classifier = nn.Identity()
bb = m.cuda().eval()
tf = transforms.Compose([transforms.Resize((160,160)), transforms.ToTensor()])
X = []
with torch.no_grad():
    for it in valid:
        try:
            x = tf(Image.open(it[0]).convert("RGB")).unsqueeze(0).cuda()
            X.append(m.features(x).flatten(1).cpu().numpy()[0])
        except Exception:
            X.append(np.zeros(1536, np.float32))
X = np.array(X, np.float32)
np.save(r"B:\mode\style_train\feats_enb3_v17.npy", X)
print("特征:", X.shape, flush=True)
STYLES = ['casual','formal','street','vintage','minimalist','sporty','boho','elegant']
SCENES = ['work','casual','party','date','home','travel']
Ys = np.array([[1.0 if s in it[1] else 0. for s in STYLES] for it in valid], np.float32)
Ye = np.array([[1.0 if s in it[2] else 0. for s in SCENES] for it in valid], np.float32)

def run(Xa, balanced, eps=400, seed=123):
    torch.manual_seed(seed)
    n_va = 250
    Xt = torch.tensor(Xa[:-n_va]).cuda(); Xv = torch.tensor(Xa[-n_va:]).cuda()
    Yst = torch.tensor(Ys[:-n_va]).cuda(); Ysv = torch.tensor(Ys[-n_va:]).cuda()
    Yet = torch.tensor(Ye[:-n_va]).cuda(); Yev = torch.tensor(Ye[-n_va:]).cuda()
    class H(nn.Module):
        def __init__(s, fin):
            super().__init__()
            s.trunk = nn.Sequential(nn.Linear(fin, 384), nn.ReLU())
            s.hs = nn.Linear(384, 8); s.he = nn.Linear(384, 6)
        def forward(s, x):
            h = s.trunk(x); return s.hs(h), s.he(h)
    net = H(Xa.shape[1]).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=eps)
    hist = {"s": [], "e": []}
    best = 0
    for ep in range(eps):
        net.train(); opt.zero_grad()
        ps, pe = net(Xt)
        ls = nn.functional.binary_cross_entropy_with_logits(ps, Yst)
        le = nn.functional.binary_cross_entropy_with_logits(pe, Yet)
        if balanced and ep > 20:
            ws = float(np.clip(np.mean(hist["e"][-20:])/max(np.mean(hist["s"][-20:]),1e-6), 0.5, 2.0))
            loss = ls*ws + le
        else:
            loss = ls + le
        hist["s"].append(ls.item()); hist["e"].append(le.item())
        loss.backward(); opt.step(); sched.step()
        if (ep+1) % 100 == 0:
            net.eval()
            with torch.no_grad():
                ps, pe = net(Xv)
                sr = (((torch.sigmoid(ps)>0.5).float()==1)&(Ysv==1)).sum().item()/max((Ysv==1).sum().item(),1)
                er = (((torch.sigmoid(pe)>0.5).float()==1)&(Yev==1)).sum().item()/max((Yev==1).sum().item(),1)
            best = max(best, sr+er)
    return best
base = run(X, False)
bal = run(X, True)
print(f"v16口径基线 sum={base:.3f}", flush=True)
print(f"v17负载均衡 sum={bal:.3f}", flush=True)
print(f"判定: 均衡{'+' if bal>base else ''}{(bal-base)*100:.1f}pt -> {'采用' if bal>base else '不采用'}", flush=True)
print("V17_DONE", flush=True)
