# Glow v8:合成配对监督(668对,同图素颜/妆后)
import glob, os
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

A = sorted(glob.glob(r"B:\mode\glow_data\pairs\*_a.jpg"))
B = sorted(glob.glob(r"B:\mode\glow_data\pairs\*_b.jpg"))
pairs = list(zip(A, B))
print(f"配对: {len(pairs)}", flush=True)

class PD(Dataset):
    def __len__(s): return len(pairs)
    def __getitem__(s, i):
        def load(p):
            return torch.tensor(np.asarray(Image.open(p).convert("RGB").resize((192,192)))/127.5-1, dtype=torch.float32).permute(2,0,1)
        a, b = pairs[i % len(pairs)]
        x, y = load(a), load(b)
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2]); y = torch.flip(y, dims=[2])
        return x, y

class GlowNet(nn.Module):
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
tr = DataLoader(PD(), batch_size=24, shuffle=True)
va_x, va_y = PD().__getitem__(5)
opt = torch.optim.AdamW(net.parameters(), lr=2e-4)
for ep in range(40):
    net.train()
    tot = 0
    for x, y in tr:
        out = net(x.cuda())
        loss = ((out - y.cuda())**2).mean()  # 纯配对监督(真目标)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    if (ep+1) % 10 == 0:
        torch.save(net.state_dict(), r"B:\mode\models\vyton_glow_v8.pt")
        print(f"ep{ep+1} loss={tot/len(tr):.4f}", flush=True)
net.eval()
with torch.no_grad():
    o = net(va_x.unsqueeze(0).cuda())[0].cpu()
grid = torch.cat([va_x, o, va_y], 2)
g = ((grid.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype("uint8")
Image.fromarray(g).save(r"B:\mode\glow_data\glow_v8_test.png")
try:
    import sys
    sys.path.insert(0, r"B:\mode\rsi")
    from rsi_hook import rsi_log
    rsi_log("glow", "v8_配对监督", f"loss{tot/len(tr):.4f}", "合成配对668对", "GREEN:配对监督替代非配对域对齐")
except Exception:
    pass
print("GLOW_V8_DONE", flush=True)
