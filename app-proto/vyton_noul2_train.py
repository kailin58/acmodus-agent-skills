# Noul质量门v2
import glob, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
pos = glob.glob(r"B:\mode\glow_gallery\*.png")
neg_pool = glob.glob(r"B:\mode\vton_training\samples\v*.png") + glob.glob(r"B:\mode\vton_training\teacher_pairs\*.png")
random.seed(3)
neg = random.sample(neg_pool, min(26, len(neg_pool)))
class DS(Dataset):
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, y = s.items[i]
        img = Image.open(p).convert("RGB").resize((160,160))
        x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1)
        if torch.rand(1).item()<0.5: x = torch.flip(x, dims=[2])
        return x, y
    def __init__(s, items): s.items = items
items = [(p,1) for p in pos] + [(p,0) for p in neg]
random.shuffle(items)
sp = int(len(items)*0.8)
tr, va = DS(items[:sp]), DS(items[sp:])
class QNet(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.net = nn.Sequential(blk(3,32), blk(32,64), blk(64,96), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(96,32); s.hd = nn.Linear(32,1)
    def forward(s,x): return s.hd(torch.relu(s.fc(s.net(x).flatten(1))))
m = QNet().cuda()
opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
lf = nn.BCEWithLogitsLoss()
for ep in range(16):
    m.train()
    for x, y in DataLoader(tr, batch_size=8, shuffle=True):
        loss = lf(m(x.cuda()), y.float().cuda().view(-1,1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); c=t=0
    with torch.no_grad():
        for x, y in DataLoader(va, batch_size=8):
            p = (torch.sigmoid(m(x.cuda()))>0.5).float().cpu().view(-1)
            c += (p==y.float()).sum().item(); t += len(y)
    print(f"ep{ep+1} val={c/max(t,1):.2f}", flush=True)
torch.save(m.state_dict(), r"B:\mode\models\vyton_glow_noul_v2.pt")
print("NOUL2_DONE", flush=True)
