# VytonSR-R2:高频重建头训练(lo→hi)
# 架构:MobileNetV3-small编码器+轻上采样(~0.5MB int8)
import torch, torch.nn as nn
import numpy as np, glob, os, time
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class PD(Dataset):
    def __init__(s, lo_dir, hi_dir):
        s.fs = sorted(glob.glob(lo_dir + "/*.png"))
        s.hi_dir = hi_dir
    def __len__(s): return len(s.fs)
    def __getitem__(s, i):
        lo = np.asarray(Image.open(s.fs[i]).convert("RGB").resize((256,256)))/127.5-1
        hi = np.asarray(Image.open(s.hi_dir + "/" + os.path.basename(s.fs[i])).convert("RGB").resize((256,256)))/127.5-1
        x = torch.tensor(lo, dtype=torch.float32).permute(2,0,1)
        y = torch.tensor(hi, dtype=torch.float32).permute(2,0,1)
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, [2]); y = torch.flip(y, [2])
        return x, y

class RebuildNet(nn.Module):
    """轻量重建头:3层残差+上采样保尺寸"""
    def __init__(s, ch=32):
        super().__init__()
        s.head = nn.Conv2d(3, ch, 3, 1, 1)
        s.body = nn.Sequential(*[ResBlk(ch) for _ in range(4)])
        s.tail = nn.Conv2d(ch, 3, 3, 1, 1)
    def forward(s, x):
        return x + s.tail(s.body(s.head(x)))

class ResBlk(nn.Module):
    def __init__(s, ch):
        super().__init__()
        s.c1 = nn.Conv2d(ch, ch, 3, 1, 1); s.a = nn.PReLU()
        s.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
    def forward(s, x):
        return x + s.c2(s.a(s.c1(x)))

BASE = r"B:/mode/models/sr/pairs"
ds = PD(BASE + "/lo", BASE + "/hi")
print("pairs", len(ds), flush=True)
n_va = 40
tr, va = torch.utils.data.random_split(ds, [len(ds)-n_va, n_va], generator=torch.Generator().manual_seed(7))
dl = DataLoader(tr, batch_size=16, shuffle=True, num_workers=0, drop_last=True)

net = RebuildNet().cuda()
opt = torch.optim.AdamW(net.parameters(), lr=2e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000)
best = 0.0
step = 0
t0 = time.time()
for ep in range(60):
    for x, y in dl:
        x, y = x.cuda(), y.cuda()
        out = net(x)
        # 感知权重:高频区(边缘)误差加倍
        hi_f = (y - torch.nn.functional.avg_pool2d(y, 3, 1, 1)).abs().mean(1, keepdim=True)
        w = 1 + 3*hi_f
        loss = ((out-y).abs()*w).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
        if step % 100 == 0:
            print(f"step{step} loss={loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    if (ep+1) % 10 == 0:
        net.eval()
        psnrs = []
        with torch.no_grad():
            for k in range(n_va):
                x, y = va[k]
                out = net(x.unsqueeze(0).cuda())[0].cpu()
                mse = ((out-y).numpy()*127.5)**2
                psnrs.append(10*np.log10(255*255/max(mse.mean(), 1e-6)))
        p = float(np.mean(psnrs))
        print(f"ep{ep+1} PSNR={p:.2f}dB", flush=True)
        if p > best:
            best = p
            torch.save(net.state_dict(), r"B:/mode/models/sr/rebuild_head.pt")
        net.train()
print(f"R2_DONE best={best:.2f}dB params={sum(p.numel() for p in net.parameters())/1e6:.2f}M")
