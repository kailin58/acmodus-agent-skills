# 虚拟自己v15:身份条件头(自训对比学习,增广正对)
import glob, os, random
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

# 底座:轻量身份编码器(自研,不做ArcFace复刻)
class IDNet(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,2,1), nn.BatchNorm2d(o), nn.ReLU())
        s.net = nn.Sequential(blk(3,32), blk(32,64), blk(64,128), blk(128,256), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(256, 128)
    def forward(s, x): return F.normalize(s.fc(s.net(x).flatten(1)), dim=1)

class PosDS(Dataset):
    """增广正对:同图两种增广=同人身份不变"""
    def __init__(s, paths): s.paths = paths
    def __len__(s): return len(s.paths)
    def __getitem__(s, i):
        img = Image.open(s.paths[i % len(s.paths)]).convert("RGB")
        W, H = img.size
        img = img.crop((0, 0, W, int(H*0.62))).resize((112,112))
        x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1)
        def aug(t):
            if torch.rand(1).item()<0.5: t = torch.flip(t, dims=[2])
            t = t * (0.8 + torch.rand(1).item()*0.4)
            # 随机裁剪平移
            jx, jy = random.randint(-6,6), random.randint(-4,4)
            t = torch.roll(t, (jy, jx), dims=(1,2))
            return t.clamp(0,1)
        return aug(x), aug(x)

paths = sorted(glob.glob(r"B:\mode\vton_training\teacher_pairs_v2\t_0*.png")) + \
        sorted(glob.glob(r"B:\mode\ethnic_data\*\*.jpg"))
print("身份数据:", len(paths), flush=True)
ds = PosDS(paths)
dl = DataLoader(ds, batch_size=64, shuffle=True)
net = IDNet().cuda()
opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
temp = 0.15
for ep in range(8):
    tot=0; nb=0
    for a, b in dl:
        a, b = a.cuda(), b.cuda()
        za, zb = net(a), net(b)
        # InfoNCE(批内负对)
        logits = za @ zb.T / temp
        labels = torch.arange(len(a), device="cuda")
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    print(f"ep{ep+1} loss={tot/nb:.4f}", flush=True)
torch.save(net.state_dict(), r"B:\mode\models\vyton_id_head.pt")
print("ID_HEAD_DONE", flush=True)
