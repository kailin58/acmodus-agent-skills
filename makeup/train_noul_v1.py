
# Vyton-Glow质感判别头(Noul):输入妆容图→质量分(0-1)
import os, glob
import numpy as np
import torch, torch.nn as nn
from PIL import Image

pos = glob.glob(r"B:\mode\glow_gallery\*.png")[:20]
neg_pool = glob.glob(r"B:\mode\vton_training\samples\v2_*.png")[:20]
print("pos:", len(pos), "neg:", len(neg_pool), flush=True)

class DS(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB").resize((128,128))
        return torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1), y

items = [(p, 1) for p in pos] + [(p, 0) for p in neg_pool]
ds = DS(items)
n = len(ds); sp = int(n*0.8)
tr, va = torch.utils.data.random_split(ds, [sp, n-sp])

class QNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(3,32,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,96,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(96,32); s.hd = nn.Linear(32,1)
    def forward(s,x): return s.hd(torch.relu(s.fc(s.net(x).flatten(1))))

m = QNet().cuda()
opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
lf = nn.BCEWithLogitsLoss()
for ep in range(12):
    m.train()
    for x, y in torch.utils.data.DataLoader(tr, batch_size=8, shuffle=True):
        loss = lf(m(x.cuda()), y.float().cuda().view(-1,1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); c = t = 0
    with torch.no_grad():
        for x, y in torch.utils.data.DataLoader(va, batch_size=8):
            p = (torch.sigmoid(m(x.cuda()))>0.5).float().cpu().view(-1)
            c += (p==y.float()).sum().item(); t += len(y)
    print(f"ep{ep+1} val={c/max(t,1):.2f}", flush=True)
torch.save(m.state_dict(), r"B:\mode\models\vyton_glow_noul_v1.pt")
print("NOUL_SAVED", flush=True)
