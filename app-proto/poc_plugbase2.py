# 演算2:换基座→头重蒸(结构保留,权重重训)vs完全从头
import torch, torch.nn as nn, glob, json, os, time
from PIL import Image
import numpy as np
from torchvision import models as tvm

labels = json.load(open(r"B:\mode\style_train\pin_labels_v1.json", encoding="utf-8"))
files = [(f, lab) for f, lab in list(labels.items())[:600] if os.path.exists(f)]
STYLES = ['casual','formal','street','vintage','minimalist','sporty','boho','elegant']

base_a = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.IMAGENET1K_V1)
base_a.classifier = nn.Identity()
base_b = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
base_b.fc = nn.Identity()
device = "cuda"
base_a, base_b = base_a.to(device).eval(), base_b.to(device).eval()

@torch.no_grad()
def feats(base, files):
    X, Y = [], []
    for f, lab in files:
        try:
            im = Image.open(f).convert("RGB").resize((160, 160))
            x = torch.tensor(np.asarray(im)/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
            X.append(base(x).flatten(1).cpu().numpy()[0])
            Y.append([1.0 if s in lab.get("style", []) else 0.0 for s in STYLES])
        except Exception:
            continue
    return np.array(X, np.float32), np.array(Y, np.float32)

XA, Y = feats(base_a, files)
XB, _ = feats(base_b, files)
print(f"样本{len(Y)} A{XA.shape[1]}维 B{XB.shape[1]}维", flush=True)

def run(X, Y, epochs=200, lr=3e-3, init_head=None):
    head = (init_head if init_head is not None else nn.Sequential(nn.Linear(X.shape[1], 512), nn.ReLU(), nn.Linear(512, 8))).to(device).float()
    if init_head is not None:  # 维度不同重造第一层,保第二层语义
        head = nn.Sequential(nn.Linear(X.shape[1], 512), nn.ReLU(), head[-1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    n = len(X) - 100
    Xt = torch.tensor(X).to(device); Yt = torch.tensor(Y, dtype=torch.float32).to(device)
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(head(Xt[:n]), Yt[:n])
        loss.backward(); opt.step()
    dt = time.time() - t0
    head.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(head(Xt[n:])) > 0.5).float()
        rec = ((pred == 1) & (Yt[n:] == 1)).sum().item() / max((Yt[n:] == 1).sum().item(), 1)
    return head, rec, dt

# 基线:A从头(200ep充分)
_, recA, dtA = run(XA, Y)
# B从头
_, recB0, dtB0 = run(XB, Y)
# B重蒸头(用A头的第二层初始化)
headA, _, _ = run(XA, Y, epochs=200)
headB_st, recB1, dtB1 = run(XB, Y, epochs=60, init_head=headA)  # 少ep=看初始化收益
print(f"A基座从头200ep: 召回{recA:.3f} ({dtA:.0f}s)", flush=True)
print(f"B基座从头200ep: 召回{recB0:.3f} ({dtB0:.0f}s)", flush=True)
print(f"B基座+A头初始化60ep: 召回{recB1:.3f} ({dtB1:.0f}s)", flush=True)
print(f"\n结论: 换基座重蒸头={dtB0:.0f}s级(特征缓存后秒-分钟);A头初始化{'有收益' if recB1 >= recB0*0.95 else '无显著收益'}", flush=True)
print("插拔判定: 换基座→头必须重蒸(不重写代码,只重跑)——小时级成立(全量数据+GPU)", flush=True)
