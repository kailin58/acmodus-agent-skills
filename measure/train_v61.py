# Vyton-Measure v6.1:17解剖点+8围度带→6维
import json
import numpy as np
import torch, torch.nn as nn

data = json.load(open(r"B:\mode\measure\syn_kps_v2\data.json", encoding="utf-8"))
print("数据:", len(data), flush=True)
KEYS = ["height","weight","gender","muscle","age","proportions"]
X = np.array([np.array(d["kp"], dtype=np.float32).reshape(-1).tolist() + np.array(d["wd"], dtype=np.float32).tolist() for d in data], dtype=np.float32)
Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
Y = np.array([[d[k] for k in KEYS] for d in data], dtype=np.float32)
n = len(Xn); sp = int(n*0.9)
xt = torch.tensor(Xn[:sp]); yt = torch.tensor(Y[:sp]); xv = torch.tensor(Xn[sp:]); yv = torch.tensor(Y[sp:])

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(59, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 6))
    def forward(s, x): return s.net(x)

m = Net().cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
best = 9
for ep in range(40):
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
    sc = mae[0]+mae[1]+mae[3]
    print(f"ep{ep+1} h={mae[0]:.5f} w={mae[1]:.5f} g={mae[2]:.4f} mu={mae[3]:.4f} ag={mae[4]:.4f} pr={mae[5]:.5f}", flush=True)
    if sc < best:
        best = sc
        torch.save(m.state_dict(), r"B:\mode\models\vyton_measure_v61.pt")
    sch.step()
print("MV61_DONE", round(best,5), flush=True)
