
# Vyton-Measure v3:6维尺寸回归(身高/体重/肩宽/臀宽/躯干/腿长)
import json
import numpy as np
import torch
import torch.nn as nn

data = json.load(open(r"B:\mode\measure\landmarks_full.json", encoding="utf-8"))
X, Y = [], []
for rec in data:
    lm3 = np.array(rec["kps"], dtype=np.float32).reshape(-1)
    lm = lm3.reshape(33,3)[:,:2] if lm3.size==99 else lm3.reshape(33,2)
    lm = np.array(lm, dtype=np.float32)
    c = (lm[23]+lm[24])/2
    span = np.linalg.norm(lm[10]-lm[31]) + 1e-6
    X.append(((lm-c)/span).reshape(-1))
    Y.append([rec["height"], rec["weight"], rec["shoulder_m"], rec["hip_m"], rec["torso_m"], rec["leg_m"]])
X = torch.tensor(np.array(X))
Y = torch.tensor(np.array(Y), dtype=torch.float32)
# 目标归一(0-1内已基本满足,保险clip)
Y = torch.clamp(Y, 0, 1)

class M3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(66,160), nn.ReLU(), nn.Linear(160,96), nn.ReLU())
        self.head = nn.Linear(96,6)
    def forward(self,x): return self.head(self.net(x))

torch.manual_seed(0)
m = M3()
opt = torch.optim.AdamW(m.parameters(), lr=1.2e-3)
lf = nn.MSELoss()
n = len(X); split = int(n*0.85)
best_val = 9
for ep in range(600):
    m.train(); opt.zero_grad()
    pred = m(X[:split])
    loss = lf(pred, Y[:split])
    loss.backward(); opt.step()
    if (ep+1)%150==0:
        m.eval()
        with torch.no_grad():
            vp = m(X[split:])
            vl = lf(vp, Y[split:]).item()
        print(f"ep{ep+1} train={loss.item():.5f} val={vl:.5f}", flush=True)
        if vl < best_val:
            best_val = vl
            torch.save(m.state_dict(), r"B:\mode\models\vyton_measure_v3.pt")
# 分维度误差(相对)
m.eval()
with torch.no_grad():
    vp = m(X[split:]).numpy()
yt = Y[split:].numpy()
names = ["身高","体重","肩宽","臀宽","躯干","腿长"]
for i, nm in enumerate(names):
    mae = np.abs(vp[:,i]-yt[:,i]).mean()
    print(f"{nm}: MAE={mae:.4f}", flush=True)
print("M3_DONE best_val:", round(best_val,5), flush=True)
