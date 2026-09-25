
import os, io, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pyarrow.parquet as pq
import glob

# 数据:mesh_50渲染图+VITON高清原图→(降质512, 原1024)对
class SRData(Dataset):
    def __init__(self):
        self.pairs = []
        pf = pq.ParquetFile(r"B:\\mode\\vton_training\\data\\VITON-HD-Captions\\data\\train-00000-of-00012.parquet")
        t = pf.read_row_group(0)
        for i in range(min(220, t.num_rows)):
            b = t.column('image')[i].as_py()['bytes']
            img = Image.open(io.BytesIO(b)).convert("RGB")
            if min(img.size) < 512: continue
            self.pairs.append(b)
        print("pairs:", len(self.pairs), flush=True)
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        img = Image.open(io.BytesIO(self.pairs[i])).convert("RGB")
        w, h = img.size
        img = img.crop((0, 0, w - w%8, h - h%8)).resize((256, 256))
        hi = np.array(img, dtype=np.float32)/255.0
        lo = np.array(img.resize((64,64), Image.BICUBIC).resize((256,256), Image.BICUBIC), dtype=np.float32)/255.0
        return torch.tensor(lo).permute(2,0,1), torch.tensor(hi).permute(2,0,1)

class ResBlock(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.b = nn.Sequential(nn.Conv2d(c,c,3,1,1), nn.ReLU(), nn.Conv2d(c,c,3,1,1))
    def forward(s, x): return x + s.b(x)

class TinySR(nn.Module):
    def __init__(s):
        super().__init__()
        s.head = nn.Conv2d(3, 48, 3, 1, 1)
        s.body = nn.Sequential(*[ResBlock(48) for _ in range(6)])
        s.tail = nn.Conv2d(48, 3, 3, 1, 1)
    def forward(s, x): return x + s.tail(s.body(s.head(x)))

ds = SRData()
n = len(ds); sp = int(n*0.95)
tr, va = torch.utils.data.random_split(ds, [sp, n-sp])
m = TinySR().cuda()
opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
lf = nn.L1Loss()
for ep in range(3):
    m.train()
    for lo, hi in DataLoader(tr, batch_size=16, shuffle=True):
        loss = lf(m(lo.cuda()), hi.cuda())
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); tot = 0.0; cnt = 0
    with torch.no_grad():
        for lo, hi in DataLoader(va, batch_size=8):
            tot += lf(m(lo.cuda()), hi.cuda()).item(); cnt += 1
    print(f"ep{ep+1} val_L1={tot/max(cnt,1):.4f}", flush=True)
torch.save(m.state_dict(), r"B:\\mode\\models\\vyton_sr_v1.pt")
print("SR_SAVED", flush=True)
