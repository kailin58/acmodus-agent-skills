# Try学生v6:半身自监督扩散小UNet(600教师图,质量门过滤)
import glob, os
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def black_block_score(arr):
    """启发式:大块纯黑区占比(>12%判崩)"""
    g = arr.mean(axis=2)
    dark = (g < 18).mean()
    return dark

paths = sorted(glob.glob(r"B:\mode\vton_training\teacher_pairs_v2\t_*.png"))
good = []
for p in paths:
    a = np.asarray(Image.open(p).convert("RGB").resize((128,128)))
    if black_block_score(a) < 0.08:
        good.append(p)
print("质量门:", len(good), "/", len(paths), flush=True)

class DS(Dataset):
    def __init__(s, paths):
        s.paths = paths
    def __len__(s): return len(s.paths)
    def __getitem__(s, i):
        img = Image.open(s.paths[i % len(s.paths)]).convert("RGB")
        W, H = img.size
        # 半身裁剪(上60%)
        img = img.crop((0, 0, W, int(H*0.62))).resize((128, 128))
        x = torch.tensor(np.asarray(img)/127.5-1, dtype=torch.float32).permute(2,0,1)
        if torch.rand(1).item()<0.5: x = torch.flip(x, dims=[2])
        return x

class UNet(nn.Module):
    def __init__(s, ch=48):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.GroupNorm(8,o), nn.SiLU(), nn.Conv2d(o,o,3,padding=1), nn.GroupNorm(8,o), nn.SiLU())
        s.d1 = blk(3+1, ch); s.d2 = blk(ch, ch*2); s.d3 = blk(ch*2, ch*3)
        s.mid = blk(ch*3, ch*3)
        s.u3 = blk(ch*3+ch*3, ch*2); s.u2 = blk(ch*2+ch*2, ch); s.u1 = blk(ch+ch, ch)
        s.out = nn.Conv2d(ch, 3, 1)
        s.pool = nn.MaxPool2d(2)
    def forward(s, x, t_emb):
        t = t_emb.view(-1,1,1,1).expand(-1,1,x.shape[2],x.shape[3])
        h1 = s.d1(torch.cat([x,t],1)); h2 = s.d2(s.pool(h1)); h3 = s.d3(s.pool(h2))
        m = s.mid(s.pool(h3))
        u3 = s.u3(torch.cat([nn.functional.interpolate(m, scale_factor=2, mode="nearest"), h3],1))
        u2 = s.u2(torch.cat([nn.functional.interpolate(u3, scale_factor=2, mode="nearest"), h2],1))
        u1 = s.u1(torch.cat([nn.functional.interpolate(u2, scale_factor=2, mode="nearest"), h1],1))
        return s.out(u1)

dev = "cuda"
ds = DS(good)
dl = DataLoader(ds, batch_size=24, shuffle=True)
net = UNet().to(dev)
print("参数:", sum(p.numel() for p in net.parameters())//1000, "K", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=1.2e-4)
T = 400
betas = torch.linspace(1e-4, 0.02, T).to(dev)
alphas = 1-betas
abar = torch.cumprod(alphas, 0).to(dev)

def add_noise(x0, t):
    a = abar[t].view(-1,1,1,1)
    nz = torch.randn_like(x0)
    return (a.sqrt()*x0 + (1-a).sqrt()*nz), nz

for ep in range(600):
    tot = 0; nb = 0
    for x in dl:
        x = x.to(dev)
        t = torch.randint(0, T, (len(x),), device=dev)
        xt, nz = add_noise(x, t)
        pred = net(xt, t.float()/T)
        loss = nn.functional.mse_loss(pred, nz)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    print(f"ep{ep+1} loss={tot/nb:.4f}", flush=True)
torch.save(net.state_dict(), r"B:\mode\models\vyton_try_student_v6c.pt")
print("TRY_STUDENT_V6C_DONE", flush=True)
