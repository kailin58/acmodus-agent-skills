
# 测量头v2:33关键点→(height,weight)回归+性别分类头(三任务联合)
import os, json
import numpy as np
import torch
import torch.nn as nn

data = json.load(open(r"B:\\mode\\measure\\landmarks_full.json", encoding="utf-8"))
print("数据:", len(data), flush=True)
X, Y, G = [], [], []
for rec in data:
    lm = np.array(rec["kps"], dtype=np.float32).reshape(-1)[:66]  # 33x2
    # 归一化(以腰中心为原点,身高为单位)
    lm2 = lm.reshape(33,2)
    c = (lm2[23]+lm2[24])/2
    span = np.linalg.norm(lm2[10]-lm2[31:32][0]) + 1e-6
    lm = ((lm2-c)/span).reshape(-1)
    X.append(lm)
    Y.append([rec.get("height",0.5), rec.get("weight",0.5)])
    sh = np.linalg.norm(lm2[11]-lm2[12])+1e-6
    hp = np.linalg.norm(lm2[23]-lm2[24])+1e-6
    G.append(0 if sh/hp > 1.3 else 1)  # 0男1女(启发式标签示范,正式版人工标)
X = torch.tensor(np.array(X)); Y = torch.tensor(np.array(Y), dtype=torch.float32)
G = torch.tensor(np.array(G), dtype=torch.long)

class MeasureHeadV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(66,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU())
        self.reg = nn.Linear(64,2)     # height,weight
        self.gen = nn.Linear(64,2)     # 男/女
    def forward(self, x):
        f = self.backbone(x)
        return self.reg(f), self.gen(f)

torch.manual_seed(0)
m = MeasureHeadV2()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
lossf_reg = nn.MSELoss(); lossf_cls = nn.CrossEntropyLoss()
n = len(X); split = int(n*0.85)
for ep in range(400):
    m.train(); opt.zero_grad()
    yr, yc = m(X[:split])
    loss = lossf_reg(yr, Y[:split]) + 0.3*lossf_cls(yc, G[:split])
    loss.backward(); opt.step()
    if (ep+1) % 100 == 0:
        m.eval()
        with torch.no_grad():
            vr, vc = m(X[split:])
            vr_loss = lossf_reg(vr, Y[split:]).item()
            acc = (vc.argmax(1) == G[split:]).float().mean().item()
        print(f"ep{ep+1} train={loss.item():.4f} val_reg={vr_loss:.4f} gen_acc={acc:.2f}", flush=True)
torch.save(m.state_dict(), r"B:\\mode\\measure\\measure_head_v2.pt")
print("V2_SAVED", flush=True)
