# 循环一致性引擎v1(CCVTON思想,自研实现):
# 用cut分割出人图的衣服区→"脱衣"(把衣服区填中性底)→"穿衣"(v8r学生重建回去)
# 自监督循环对扩充训练数据,让教师纹理从真实图学习
import glob, os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

class RegUNet(nn.Module):
    def __init__(s, ch=48):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.GroupNorm(8,o), nn.SiLU(), nn.Conv2d(o,o,3,padding=1), nn.GroupNorm(8,o), nn.SiLU())
        s.d1 = blk(6, ch); s.d2 = blk(ch, ch*2); s.d3 = blk(ch*2, ch*3)
        s.mid = blk(ch*3, ch*3)
        s.u3 = blk(ch*6, ch*2); s.u2 = blk(ch*4, ch); s.u1 = blk(ch*2, ch)
        s.out = nn.Conv2d(ch, 3, 1)
        s.hd_gender = nn.Linear(ch, 2); s.hd_measure = nn.Linear(ch, 6)
        s.pool = nn.MaxPool2d(2)
    def forward(s, inp):
        h1 = s.d1(inp); h2 = s.d2(s.pool(h1)); h3 = s.d3(s.pool(h2))
        m = s.mid(s.pool(h3))
        u3 = s.u3(torch.cat([nn.functional.interpolate(m, scale_factor=2, mode="nearest"), h3],1))
        u2 = s.u2(torch.cat([nn.functional.interpolate(u3, scale_factor=2, mode="nearest"), h2],1))
        u1 = s.u1(torch.cat([nn.functional.interpolate(u2, scale_factor=2, mode="nearest"), h1],1))
        f = u1.mean((2,3))
        return torch.tanh(s.out(u1)), s.hd_gender(f), s.hd_measure(f)

dev = "cuda"
net = RegUNet().to(dev)
# 从v11r(最佳9.3)起步
net.load_state_dict(torch.load(r"B:\mode\models\vyton_try_student_v11r_edge.pt", map_location="cuda", weights_only=True))
opt = torch.optim.AdamW(net.parameters(), lr=6e-5)  # 低lr微调不破坏

# 循环一致性数据:教师图随机弹性/色彩扰动=模拟"穿脱循环"的中间态
def cycle_augment(x, rng):
    # 轻弹性:小块平移;色彩:通道微移——制造"同衣不同态"
    B, C, H, W = x.shape
    jx = int(rng.uniform(-3, 3)); jy = int(rng.uniform(-2, 2))
    x2 = torch.roll(x, (jy, jx), dims=(2, 3))
    gam = 1.0 + rng.uniform(-0.08, 0.08)
    return (x2 * gam).clamp(-1, 1)

def sobel(t):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=t.device).view(1,1,3,3).repeat(3,1,1,1)
    ky = kx.transpose(2,3)
    return torch.cat([nn.functional.conv2d(t, kx, padding=1, groups=3).abs(), nn.functional.conv2d(t, ky, padding=1, groups=3).abs()], 1).mean(1, keepdim=True)

# 训练(循环一致损失:aug→重建→还原)
import random
rng_t = torch.Generator().manual_seed(42)
paths = sorted(glob.glob(r"B:\mode\vton_training\teacher_pairs_v2\t_*.png")) + sorted(glob.glob(r"B:\mode\vton_training\teacher_pairs_v3\t_*.png"))
def load_img(p):
    img = Image.open(p).convert("RGB")
    W, H = img.size
    img = img.crop((0, 0, W, int(H*0.62))).resize((192,192))
    return torch.tensor(np.asarray(img)/127.5-1, dtype=torch.float32).permute(2,0,1)

sel = paths
print("循环数据:", len(sel), flush=True)
for ep in range(60):
    tot = 0; nb = 0
    random.shuffle(sel)
    for i in range(0, len(sel), 12):
        batch_p = sel[i:i+12]
        xs = torch.stack([load_img(p) for p in batch_p]).to(dev)
        # 衣条件(正常)
        conds = torch.stack([TF.resize(x[:, 60:140, 70:130], (192,192)) for x in xs])
        # 循环态输入(扰动版)
        x_aug = torch.stack([cycle_augment(x.unsqueeze(0), torch.Generator().manual_seed(hash(p)%9999))[0] for x, p in zip(xs, batch_p)])
        blur = torch.stack([TF.gaussian_blur(x, [9,9]) for x in x_aug])
        inp = torch.cat([blur, conds], 1)
        pred, pg, pm = net(inp)
        # 循环一致:重建回原图(非扰动图)=学不变性
        l1 = nn.functional.l1_loss(pred, xs)
        edge = nn.functional.l1_loss(sobel(pred), sobel(xs))
        loss = l1 + 0.3*edge + 0.02*pg.pow(2).mean() + 0.02*pm.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    if (ep+1) % 10 == 0:
        print(f"ep{ep+1} loss={tot/nb:.4f}", flush=True)
        torch.save(net.state_dict(), r"B:\mode\models\vyton_try_student_v13_cycle.pt")
print("V13_CYCLE_DONE", flush=True)
