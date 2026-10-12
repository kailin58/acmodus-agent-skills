# cut_v2.1:320输入+边缘加权(修边缘羽化/糊边)
import torch, torch.nn as nn, glob, os, json, time
import numpy as np
from PIL import Image
from torchvision import models as tvm, transforms

BS, IN = 24, 320
class BackboneV2(nn.Module):
    def __init__(s):
        super().__init__()
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        s.feat = m.features
    def forward(s, x): return s.feat(x)
class HeadV2(nn.Module):
    def __init__(s, inch=576):
        super().__init__()
        s.up = nn.Sequential(nn.Conv2d(inch,64,1), nn.ReLU(True), nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False), nn.Conv2d(64,32,3,padding=1), nn.ReLU(True), nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False), nn.Conv2d(32,16,3,padding=1), nn.ReLU(True), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(16,1,3,padding=1))
    def forward(s, f): return s.up(f)
bb, hd = BackboneV2().cuda(), HeadV2().cuda()
bb.load_state_dict(torch.load(r"B:\mode\models\cut\vyton_cut_v2_backbone.pt"))
hd.load_state_dict(torch.load(r"B:\mode\models\cut\vyton_cut_v2_head.pt"))
net = nn.Sequential(bb, hd)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
tf_img = transforms.Compose([transforms.Resize((IN,IN)), transforms.ToTensor()])
pairs = glob.glob(r"B:\mode\models\cut\pseudo_labels\*.png")
labs = {os.path.splitext(os.path.basename(p))[0] + ".jpg": p for p in pairs}
srcs = glob.glob(r"B:\mode\pinterest\outfit_imgs\*.jpg") + glob.glob(r"B:\mode\yt_outfit\*.jpg") + glob.glob(r"B:\mode\pinterest\board_imgs\*.jpg") + glob.glob(r"B:\mode\pet_out7n\*\*.jpg")
data = [(s, labs[k]) for s in srcs if (k := os.path.splitext(os.path.basename(s))[0]) in labs]
print(f"pairs {len(data)}", flush=True)
def edge_w(m):
    gy, gx = torch.gradient(m, dim=(1,2))
    return 1 + 4*torch.exp(-((gy+gx).abs()))
best = 0
step = 0
for ep in range(30):
    np.random.shuffle(data)
    for i in range(0, len(data)-BS, BS):
        xb, yb = [], []
        for s, l in data[i:i+BS]:
            try:
                xb.append(tf_img(Image.open(s).convert("RGB")))
                yb.append(tf_img(Image.open(l).convert("L")))
            except Exception: continue
        if len(xb) < 8: continue
        x = torch.stack(xb).cuda(); y = torch.stack(yb).cuda()
        p = torch.sigmoid(net(x))
        w = edge_w(y)
        loss = nn.functional.binary_cross_entropy(p, y, weight=w)
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
    print(f"ep{ep+1} loss {loss.item():.4f}", flush=True)
    torch.save(bb.state_dict(), r"B:\mode\models\cut\vyton_cut_v21_backbone.pt")
    torch.save(hd.state_dict(), r"B:\mode\models\cut\vyton_cut_v21_head.pt")
print("V21_DONE", flush=True)
