# Try学生v8r:回归式(非扩散)——教师图自重建+衣条件,U-Net纯L1
import glob, os
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def black_block_score(arr):
    return ((arr.mean(axis=2)) < 18).mean()

tpaths = sorted(glob.glob(r"B:\mode\vton_training\teacher_pairs_v2\t_*.png")) + \
        sorted(glob.glob(r"B:\mode\vton_training\teacher_pairs_v3\t_*.png"))
good = [p for p in tpaths if black_block_score(np.asarray(Image.open(p).convert("RGB").resize((128,128)))) < 0.08]
print("教师图:", len(good), flush=True)

class RDS(Dataset):
    def __len__(s): return len(s.paths)
    def __init__(s, paths): s.paths = paths
    def __getitem__(s, i):
        img = Image.open(s.paths[i % len(s.paths)]).convert("RGB")
        W, H = img.size
        img = img.crop((0, 0, W, int(H*0.62))).resize((192,192))
        x = torch.tensor(np.asarray(img)/127.5-1, dtype=torch.float32).permute(2,0,1)
        # 输入=模糊版(模拟低信息),目标=原图(学补全细节)
        import torchvision.transforms.functional as TF
        blur = TF.gaussian_blur(x, [9,9])
        cond = TF.resize(x[:, 60:140, 70:130], (192,192))  # 衣条件
        inp = torch.cat([blur, cond], 0)
        if torch.rand(1).item()<0.5:
            inp = torch.flip(inp, dims=[2]); x = torch.flip(x, dims=[2])
        return inp, x

class RegUNet(nn.Module):
    def __init__(s, ch=48):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.GroupNorm(8,o), nn.SiLU(), nn.Conv2d(o,o,3,padding=1), nn.GroupNorm(8,o), nn.SiLU())
        s.d1 = blk(6, ch); s.d2 = blk(ch, ch*2); s.d3 = blk(ch*2, ch*3)
        s.mid = blk(ch*3, ch*3)
        s.u3 = blk(ch*6, ch*2); s.u2 = blk(ch*4, ch); s.u1 = blk(ch*2, ch)
        s.out = nn.Conv2d(ch, 3, 1)
        s.pool = nn.MaxPool2d(2)
    def forward(s, inp):
        h1 = s.d1(inp); h2 = s.d2(s.pool(h1)); h3 = s.d3(s.pool(h2))
        m = s.mid(s.pool(h3))
        u3 = s.u3(torch.cat([nn.functional.interpolate(m, scale_factor=2, mode="nearest"), h3],1))
        u2 = s.u2(torch.cat([nn.functional.interpolate(u3, scale_factor=2, mode="nearest"), h2],1))
        u1 = s.u1(torch.cat([nn.functional.interpolate(u2, scale_factor=2, mode="nearest"), h1],1))
        return torch.tanh(s.out(u1))

dev = "cuda"
ds = RDS(good)
dl = DataLoader(ds, batch_size=12, shuffle=True)
net = RegUNet().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=2e-4)
schd = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
for ep in range(300):
    tot=0; nb=0
    for inp, x in dl:
        inp, x = inp.to(dev), x.to(dev)
        pred = net(inp)
        loss = nn.functional.l1_loss(pred, x) + 0.1*nn.functional.mse_loss(pred, x)
        opt.zero_grad(); loss.backward(); opt.step()
        tot+=loss.item(); nb+=1
    schd.step()
    if (ep+1) % 20 == 0:
        print(f"ep{ep+1} loss={tot/nb:.4f}", flush=True)
        torch.save(net.state_dict(), r"B:\mode\models\vyton_try_student_v8r.pt")
print("V8R_DONE", flush=True)
