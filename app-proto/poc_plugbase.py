# 插拔式PoC:换基座(enb3→resnet18)只重蒸头,验证"小时级换基座"
import torch, torch.nn as nn, glob, json, os, time
from PIL import Image
import numpy as np
from torchvision import models as tvm

# 数据(小样本快验:300张Pinterest意图标注)
labels = json.load(open(r"B:\mode\style_train\pin_labels_v1.json", encoding="utf-8"))
files = [(f, lab) for f, lab in list(labels.items())[:300] if os.path.exists(f)]
print(f"样本: {len(files)}", flush=True)
STYLES = ['casual','formal','street','vintage','minimalist','sporty','boho','elegant']

# 头:v16现役(Spine→hd_sty),冻结权重原样迁移
class HeadV16(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fin, 512), nn.ReLU(), nn.Linear(512, 8))
    def forward(self, x): return self.net(x)

# 基座A:enb3(现役,49.7MB)
base_a = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.IMAGENET1K_V1)
fa = base_a.classifier[1].in_features
base_a.classifier = nn.Identity()
# 基座B:resnet18(替代,44MB,不同家族)
base_b = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
fb = base_b.fc.in_features
base_b.fc = nn.Identity()
print(f"基座A enb3特征{fa}维 | 基座B resnet18特征{fb}维", flush=True)

device = "cuda"
base_a, base_b = base_a.to(device).eval(), base_b.to(device).eval()

# 特征提取(两基座同图各提一份)
@torch.no_grad()
def feats(base, files):
    X, Y = [], []
    for f, lab in files:
        try:
            im = Image.open(f).convert("RGB").resize((160, 160))
            x = torch.tensor(np.asarray(im)/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
            h = base(x).flatten(1) if hasattr(base, 'fc') is False else base(x).flatten(1)
            X.append(h.cpu().numpy()[0])
            Y.append([1.0 if s in lab.get("style", []) else 0.0 for s in STYLES])
        except Exception:
            continue
    return np.array(X, np.float32), np.array(Y, np.float32)

t0 = time.time()
XA, Y = feats(base_a, files)   # A特征(enb3逐层到classifier前=1536)
XB, _ = feats(base_b, files)   # B特征(resnet18=512)
print(f"特征提取完({time.time()-t0:.0f}s): A{XA.shape} B{XB.shape}", flush=True)

# 演算1:头迁移(在A上训头→冻结→换到B上只微调 vs 从头训)
def train_head(X, Y, epochs=60, lr=1e-3):
    head = HeadV16(X.shape[1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X).to(device); Yt = torch.tensor(Y).to(device)
    n = len(Xt) - 50
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        out = head(Xt[:n])
        loss = lf(out, Yt[:n])
        loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(head(Xt[n:])) > 0.5).float()
        yt = Yt[n:]
        hit = ((pred == 1) & (yt == 1)).sum().item()
        tot = (yt == 1).sum().item()
        rec = hit / max(tot, 1)
    return head, rec

# 路线1:enb3上从头训头(基线)
_, recA = train_head(XA, Y)
print(f"基座A(enb3)+新头: 风格召回{recA:.3f}", flush=True)

# 路线2:头A迁移到B(冻结全部,只调维度适配层=1个Linear)
class Adapter(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        self.lin = nn.Linear(fin, fout)
    def forward(self, x): return self.lin(x)

headA, _ = train_head(XA, Y)
headA.eval()
adapter = Adapter(XB.shape[1], XA.shape[1]).to(device)
opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)
XtB = torch.tensor(XB).to(device); Yt = torch.tensor(Y).to(device)
n = len(XtB) - 50
for ep in range(40):  # 只训适配层40ep
    opt.zero_grad()
    out = headA(adapter(XtB[:n]))
    loss = nn.functional.binary_cross_entropy_with_logits(out, Yt[:n])
    loss.backward(); opt.step()
with torch.no_grad():
    pred = (torch.sigmoid(headA(adapter(XtB[n:]))) > 0.5).float()
    hit = ((pred == 1) & (Yt[n:] == 1)).sum().item()
    tot = (Yt[n:] == 1).sum().item()
    recB = hit / max(tot, 1)
print(f"基座B(resnet18)+v16头经适配层: 风格召回{recB:.3f}", flush=True)
print(f"\n=== 演算结论 ===", flush=True)
print(f"换基座保头(A→B):召回{recA:.2f}→{recB:.2f}", flush=True)
print(f"插拔可行性: {'✓成立(适配层恢复精度)' if recB >= recA*0.85 else '△需重蒸头(适配层不够)'}", flush=True)
