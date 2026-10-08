# Glow v6:MT数据集(384妆后+334素颜)素颜→妆后轻量回归
import glob, os
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time

MK = glob.glob(r"B:\mode\glow_data\mt\extracted\makeup\**\*.jpg", recursive=True)
NM = glob.glob(r"B:\mode\glow_data\mt\extracted\non-makeup\**\*.jpg", recursive=True)
print(f"妆后{len(MK)} 素颜{len(NM)}", flush=True)

class DS(Dataset):
    """非配对域训练:随机素颜x+随机妆后target(域级回归)"""
    def __init__(s, nm, mk):
        s.nm, s.mk = nm, mk
    def __len__(s): return max(len(s.nm), len(s.mk))
    def __getitem__(s, i):
        def load(p):
            im = Image.open(p).convert("RGB").resize((192,192))
            return torch.tensor(np.array(im)/127.5-1, dtype=torch.float32).permute(2,0,1)
        x = load(s.nm[i % len(s.nm)])
        t = load(s.mk[int(torch.randint(0, len(s.mk), (1,)))])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2]); t = torch.flip(t, dims=[2])
        return x, t

class GlowNet(nn.Module):
    """轻量单路(1M级):下采样-瓶颈-上采样+跳连"""
    def __init__(s, ch=24):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.InstanceNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, 1, 1), nn.InstanceNorm2d(o), nn.ReLU())
        s.d1 = blk(3, ch); s.d2 = blk(ch, ch*2); s.d3 = blk(ch*2, ch*3)
        s.mid = nn.Sequential(nn.Conv2d(ch*3, ch*3, 3, 1, 1), nn.ReLU(), nn.Conv2d(ch*3, ch*3, 3, 1, 1))
        def up(i, o):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(i, o, 3, 1, 1), nn.InstanceNorm2d(o), nn.ReLU())
        s.u3 = up(ch*6, ch*2); s.u2 = up(ch*4, ch); s.u1 = up(ch*2, ch)
        s.out = nn.Conv2d(ch, 3, 1)
    def forward(s, x):
        h1 = s.d1(x); h2 = s.d2(h1); h3 = s.d3(h2)
        m = s.mid(h3)
        u3 = s.u3(torch.cat([m, h3], 1))
        u2 = s.u2(torch.cat([u3, h2], 1))
        u1 = s.u1(torch.cat([u2, h1], 1))
        return torch.tanh(s.out(u1))

net = GlowNet().cuda()
n_par = sum(p.numel() for p in net.parameters())/1e6
print(f"参数{n_par:.2f}M", flush=True)
tr = DataLoader(DS(NM[:300], MK[:340]), batch_size=24, shuffle=True)
va_x, va_t = DS(NM[300:], MK[340:]).__getitem__(0)
opt = torch.optim.AdamW(net.parameters(), lr=2e-4)
for ep in range(30):
    net.train()
    tot = 0
    for x, t in tr:
        x, t = x.cuda(), t.cuda()
        out = net(x)
        # 像素+感知代理(梯度损失)
        loss = ((out-t)**2).mean() + 0.3*((out-x)**2).mean()  # 像素+保结构(妆部变化约束)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    if (ep+1) % 5 == 0:
        net.eval()
        with torch.no_grad():
            o = net(va_x.unsqueeze(0).cuda())[0].cpu()
        torch.save(net.state_dict(), r"B:\mode\models\vyton_glow_v6.pt")
        print(f"ep{ep+1} loss={tot/len(tr):.4f} 存档", flush=True)
# 出样张
net.eval()
with torch.no_grad():
    o = net(va_x.unsqueeze(0).cuda())[0].cpu()
grid = torch.cat([va_x, o, va_t], 2)
g = ((grid.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype("uint8")
Image.fromarray(g).save(r"B:\mode\glow_data\glow_v6_test.png")
print("GLOW_V6_DONE", flush=True)
