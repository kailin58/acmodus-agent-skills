# Vyton-Measure v6:4万合成关键点→6维(零噪声标签)
import json
import numpy as np
import torch, torch.nn as nn

data = json.load(open(r"B:\mode\measure\syn_kps\data.json", encoding="utf-8"))
print("数据:", len(data), flush=True)
KEYS = ["height","weight","gender","muscle","age","proportions"]

X = np.array([np.array(d["kps"], dtype=np.float32).reshape(-1) for d in data], dtype=np.float32)
# 归一化(中心+尺度)
X = (X - X.mean(0)) / (X.std(0) + 1e-6)
Y = np.array([[d[k] for k in KEYS] for d in data], dtype=np.float32)
n = len(X); sp = int(n*0.9)
Xtr, Ytr, Xva, Yva = X[:sp], Y[:sp], X[sp:], Y[sp:]
xt = torch.tensor(Xtr); yt = torch.tensor(Ytr); xv = torch.tensor(Xva); yv = torch.tensor(Yva)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(99, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 6))
    def forward(s, x): return s.net(x)

m = Net().cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
best = 9
for ep in range(30):
    m.train()
    idx = torch.randperm(len(xt))
    for i in range(0, len(xt), 1024):
        b = idx[i:i+1024]
        loss = nn.functional.smooth_l1_loss(m(xt[b].cuda()), yt[b].cuda())
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        e = (m(xv.cuda()).cpu()-yv).abs().mean(0)
    mae = e.tolist()
    sc = mae[0]+mae[1]+mae[5]
    print(f"ep{ep+1} h={mae[0]:.5f} w={mae[1]:.5f} g={mae[2]:.4f} mu={mae[3]:.4f} ag={mae[4]:.4f} pr={mae[5]:.5f}", flush=True)
    if sc < best:
        best = sc
        torch.save(m.state_dict(), r"B:\mode\models\vyton_measure_v6.pt")
    sch.step()
print("MV6_DONE", round(best,5), flush=True)
