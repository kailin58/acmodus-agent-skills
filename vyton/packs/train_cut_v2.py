# vyton_cut_v2 蒸馏训练(第一个基座/头两件套)
# 学生: MobileNetV3-small基座(frozen可选)+FPN头 → 端侧<15MB int8
import os, glob, time
import torch, torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models as tvm, transforms

# 两件套结构:
class BackboneV2(nn.Module):  # 基座:图→特征(MoE:可整体替换)
    def __init__(self):
        super().__init__()
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.feat = m.features  # 24→576通道特征图
    def forward(self, x):
        return self.feat(x)  # [B,576,H/32,W/32]

class HeadV2(nn.Module):  # 头:特征→mask(能力层,基座无关)
    def __init__(self, inch=576):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(inch, 64, 1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, 3, padding=1))  # 256→256×256 mask
    def forward(self, f):
        return self.up(f)

class CutV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BackboneV2()
        self.head = HeadV2()
    def forward(self, x):
        return self.head(self.backbone(x))

class DS(Dataset):
    def __init__(self, pairs, size=256):
        self.pairs = pairs; self.size = size
        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        src, msk = self.pairs[i]
        try:
            x = self.tf(Image.open(src).convert("RGB"))
            y = self.tf(Image.open(msk).convert("L"))
        except Exception:
            return torch.zeros(3, self.size, self.size), torch.zeros(1, self.size, self.size)
        return x, y

# 数据:伪标签配对
pseudo = glob.glob(r"B:\mode\models\cut\pseudo_labels\*.png")
pairs = []
for m in pseudo:
    name = os.path.basename(m)[:-4]  # 去.png(原名含扩展)
    for base in [r"B:\mode\pinterest\outfit_imgs", r"B:\mode\yt_outfit", r"B:\mode\bz_outfit",
                 r"B:\mode\ig_outfit\cells", r"B:\mode\pinterest\board_imgs",
                 r"B:\mode\glow_data\mt\extracted\makeup\MAKE",
                 r"B:\mode\glow_data\mt\extracted\non-makeup\non", r"B:\mode\pet\lora_data_v3"]:
        src = os.path.join(base, name)
        if os.path.exists(src):
            pairs.append((src, m)); break
print(f"配对: {len(pairs)}", flush=True)
random_split = int(len(pairs) * 0.95)
tr = DS(pairs[:random_split]); va = DS(pairs[random_split:])
trl, val = DataLoader(tr, 16, shuffle=True, num_workers=0), DataLoader(va, 16, num_workers=0)

net = CutV2().cuda()
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
bce = nn.BCEWithLogitsLoss()
# 蒸馏损失: BCE+边缘加权L1(发丝区权重x5,MoE负载均衡思想的借用)
def edge_weight(y):
    gy, gx = torch.gradient(y.squeeze(1), dim=(1, 2))
    e = (gx**2 + gy**2).clamp(0, 1)
    return 1 + e * 4  # 边缘区权重5x
best = 0
for ep in range(15):
    net.train()
    tot = 0
    for x, y in trl:
        x, y = x.cuda(), y.cuda()
        pred = net(x)
        loss = bce(pred, y) + (torch.nn.functional.l1_loss(torch.sigmoid(pred), y, reduction="none") * edge_weight(y)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    sched.step()
    # 验证:IoU
    net.eval()
    inter, union = 0, 0
    with torch.no_grad():
        for x, y in val:
            pred = (torch.sigmoid(net(x.cuda())) > 0.5).float().cpu()
            inter += (pred * y).sum().item()
            union += ((pred + y) > 0).float().sum().item()
    iou = inter / max(union, 1)
    print(f"ep{ep+1} loss{tot/len(trl):.4f} IoU{iou:.4f}", flush=True)
    if iou > best:
        best = iou
        torch.save(net.state_dict(), r"B:\mode\models\cut\vyton_cut_v2.pt")
        torch.save(net.backbone.state_dict(), r"B:\mode\models\cut\vyton_cut_v2_backbone.pt")  # 两件套:基座独立存
        torch.save(net.head.state_dict(), r"B:\mode\models\cut\vyton_cut_v2_head.pt")          # 头独立存
print(f"CUT_V2_DONE best_IoU={best:.4f}", flush=True)
