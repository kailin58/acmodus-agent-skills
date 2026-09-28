# Try学生v7:条件化UNet(衣图条件通道)——真Try-on学生
import glob, os
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def black_block_score(arr):
    g = arr.mean(axis=2)
    return (g < 18).mean()

tpaths = sorted(glob.glob(r"B:\mode\vton_training\teacher_pairs_v2\t_*.png"))
good = [p for p in tpaths if black_block_score(np.asarray(Image.open(p).convert("RGB").resize((128,128)))) < 0.08]
print("教师图:", len(good), flush=True)

class CDS(Dataset):
    """条件对:(衣图crop=条件, 全图=目标)"""
    def __init__(s, paths):
        s.paths = paths
    def __len__(s): return len(s.paths)
    def __getitem__(s, i):
        img = Image.open(s.paths[i % len(s.paths)]).convert("RGB")
        W, H = img.size
        img = img.crop((0, 0, W, int(H*0.62))).resize((128,128))
        x = torch.tensor(np.asarray(img)/127.5-1, dtype=torch.float32).permute(2,0,1)
        # 衣条件=躯干中央crop(模拟用户选的衣图)
        cx = x[:, 40:88, 44:84]
        cond = torch.nn.functional.interpolate(cx.unsqueeze(0), size=(128,128), mode="bilinear")[0]
        if torch.rand(1).item()<0.5:
            x = torch.flip(x, dims=[2]); cond = torch.flip(cond, dims=[2])
        return x, cond

class CondUNet(nn.Module):
    """条件UNet:6通道输入(噪图3+衣条件3)"""
    def __init__(s, ch=48):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.GroupNorm(8,o), nn.SiLU(), nn.Conv2d(o,o,3,padding=1), nn.GroupNorm(8,o), nn.SiLU())
        s.d1 = blk(6+1, ch); s.d2 = blk(ch, ch*2); s.d3 = blk(ch*2, ch*3)
        s.mid = blk(ch*3, ch*3)
        s.u3 = blk(ch*3+ch*3, ch*2); s.u2 = blk(ch*2+ch*2, ch); s.u1 = blk(ch+ch, ch)
        s.out = nn.Conv2d(ch, 3, 1)
        s.pool = nn.MaxPool2d(2)
    def forward(s, x, cond, t_emb):
        t = t_emb.view(-1,1,1,1).expand(-1,1,x.shape[2],x.shape[3])
        xin = torch.cat([x, cond], 1)
        h1 = s.d1(torch.cat([xin,t],1)); h2 = s.d2(s.pool(h1)); h3 = s.d3(s.pool(h2))
        m = s.mid(s.pool(h3))
        u3 = s.u3(torch.cat([nn.functional.interpolate(m, scale_factor=2, mode="nearest"), h3],1))
        u2 = s.u2(torch.cat([nn.functional.interpolate(u3, scale_factor=2, mode="nearest"), h2],1))
        u1 = s.u1(torch.cat([nn.functional.interpolate(u2, scale_factor=2, mode="nearest"), h1],1))
        return s.out(u1)

dev = "cuda"
ds = CDS(good)
dl = DataLoader(ds, batch_size=22, shuffle=True)
net = CondUNet().to(dev)
print("参数:", sum(p.numel() for p in net.parameters())//1000, "K", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=1.1e-4)
schd = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1800)
T = 400
betas = torch.linspace(1e-4, 0.02, T).to(dev)
abar = torch.cumprod(1-betas, 0).to(dev)
for ep in range(1800):
    tot=0; nb=0
    for x, cond in dl:
        x, cond = x.to(dev), cond.to(dev)
        t = torch.randint(0, T, (len(x),), device=dev)
        a = abar[t].view(-1,1,1,1)
        nz = torch.randn_like(x)
        xt = (a.sqrt()*x + (1-a).sqrt()*nz)
        pred = net(xt, cond, t.float()/T)
        loss = nn.functional.mse_loss(pred, nz)
        opt.zero_grad(); loss.backward(); opt.step()
        tot+=loss.item(); nb+=1
    schd.step()
    if (ep+1) % 100 == 0:
        print(f"ep{ep+1} loss={tot/nb:.4f}", flush=True)
        torch.save(net.state_dict(), r"B:\mode\models\vyton_try_student_v7_cond.pt")
print("V7_COND_DONE", flush=True)
torch.save(net.state_dict(), r"B:\mode\models\vyton_try_student_v7_cond.pt")
