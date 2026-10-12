# 公平对比:enb3 vs mbv3s特征+同款头
import torch, torch.nn as nn
import numpy as np, os, json, glob
from PIL import Image
from torchvision import models as tvm, transforms

X = np.load(r"B:\mode\style_train\feats_mbv3s.npy")
n = len(X)
items = []
labs = json.load(open(r"B:\mode\style_train\pin_labels_v1.json", encoding="utf-8"))
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
valid = [it for it in items if os.path.exists(it[0])][:n]
STYLES = ['casual','formal','street','vintage','minimalist','sporty','boho','elegant']
SCENES = ['work','casual','party','date','home','travel']
Ys = np.array([[1.0 if s in it[1] else 0. for s in STYLES] for it in valid], np.float32)
Ye = np.array([[1.0 if s in it[2] else 0. for s in SCENES] for it in valid], np.float32)
print(f"对齐{len(valid)}/{n}", flush=True)
m = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.IMAGENET1K_V1)
m.classifier = nn.Identity()
bb = m.cuda().eval()
tf = transforms.Compose([transforms.Resize((160,160)), transforms.ToTensor()])
Xe = []
with torch.no_grad():
    for it in valid:
        try:
            x = tf(Image.open(it[0]).convert("RGB")).unsqueeze(0).cuda()
            Xe.append(m.features(x).flatten(1).cpu().numpy()[0])
        except Exception:
            Xe.append(np.zeros(1536, np.float32))
Xe = np.array(Xe, np.float32)
print("enb3特征:", Xe.shape, flush=True)

def run_heads(Xa, Ys, Ye, eps=400):
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
    best = 0
    for ep in range(eps):
        net.train(); opt.zero_grad()
        ps, pe = net(Xt)
        loss = 0.8*nn.functional.binary_cross_entropy_with_logits(ps, Yst) + 0.8*nn.functional.binary_cross_entropy_with_logits(pe, Yet)
        loss.backward(); opt.step(); sched.step()
        if (ep+1) % 100 == 0:
            net.eval()
            with torch.no_grad():
                ps, pe = net(Xv)
                sr = (((torch.sigmoid(ps)>0.5).float()==1)&(Ysv==1)).sum().item()/max((Ysv==1).sum().item(),1)
                er = (((torch.sigmoid(pe)>0.5).float()==1)&(Yev==1)).sum().item()/max((Yev==1).sum().item(),1)
            best = max(best, sr+er)
    return best

mb = run_heads(X[:len(valid)], Ys, Ye)
eb = run_heads(Xe, Ys, Ye)
print("公平对比(同数据同头同超参):", flush=True)
print(f"mbv3s: sum={mb:.3f} (2.5M参数)", flush=True)
print(f"enb3 : sum={eb:.3f} (10.7M参数)", flush=True)
verdict = "mbv3s可替换!" if mb >= eb - 0.02 else f"enb3守住,mbv3s差{eb-mb:.3f}"
print("判定:", verdict, flush=True)
