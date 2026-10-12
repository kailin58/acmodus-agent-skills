# Style v18:数据扩容(画板新图)+enb3四头全训练
import torch, torch.nn as nn, glob, os, json
import numpy as np
from PIL import Image
from torchvision import models as tvm, transforms
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
    if st or sc: items.append((f, st, sc))
valid = [it for it in items if os.path.exists(it[0])]
print(f"items {len(valid)}", flush=True)
m = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.IMAGENET1K_V1)
m.classifier = nn.Identity()
bb = m.cuda().eval()
tf = transforms.Compose([transforms.Resize((160,160)), transforms.ToTensor()])
X, Ys_l, Ye_l = [], [], []
STYLES = ['casual','formal','street','vintage','minimalist','sporty','boho','elegant']
SCENES = ['work','casual','party','date','home','travel']
with torch.no_grad():
    for it in valid:
        try:
            x = tf(Image.open(it[0]).convert("RGB")).unsqueeze(0).cuda()
            X.append(m.features(x).flatten(1).cpu().numpy()[0])
        except Exception:
            X.append(np.zeros(1536, np.float32))
        Ys_l.append([1.0 if s in it[1] else 0. for s in STYLES])
        Ye_l.append([1.0 if s in it[2] else 0. for s in SCENES])
X = np.array(X, np.float32)
Ys = np.array(Ys_l, np.float32); Ye = np.array(Ye_l, np.float32)
print("特征", X.shape, flush=True)
n_va = 250
Xt = torch.tensor(X[:-n_va]).cuda(); Xv = torch.tensor(X[-n_va:]).cuda()
Yst = torch.tensor(Ys[:-n_va]).cuda(); Ysv = torch.tensor(Ys[-n_va:]).cuda()
Yet = torch.tensor(Ye[:-n_va]).cuda(); Yev = torch.tensor(Ye[-n_va:]).cuda()
class H(nn.Module):
    def __init__(s, fin=1536):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(fin, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 384), nn.ReLU())
        s.hs = nn.Linear(384, 8); s.he = nn.Linear(384, 6)
    def forward(s, x):
        h = s.trunk(x); return s.hs(h), s.he(h)
net = H(X.shape[1]).cuda()
opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)
best = 0
for ep in range(600):
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
        print(f"ep{ep+1} styleR{sr:.3f} sceneR{er:.3f}", flush=True)
        if sr+er > best:
            best = sr+er
            torch.save(net.state_dict(), r"B:\mode\style_train\vyton_style_v18_heads.pt")
print(f"V18_DONE best={best:.3f}", flush=True)
