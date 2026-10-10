# Try v18:轻耦合统一(RegUNet+穿搭辅助头,共享encoder)
import glob, os
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.models as models

# 数据1:换装教师对(v17配方)
TP = sorted(glob.glob(r"B:\mode\vton_training\teacher_v4c_pore\t*.png"))[:200]
TP += sorted(glob.glob(r"B:\mode\vton_training\teacher_pairs_v3\t_0*.png"))[:200]
# 数据2:穿搭图(品类粗标:Pinterest意图词含品类)
import json
labs = json.load(open(r"B:\mode\style_train\pin_labels_v1.json", encoding="utf-8"))
ITEM_MAP = {"blazer":"jacket","denim":"jacket","trench":"coat","skirt":"skirt","pants":"pants",
            "sweater":"sweater","dress":"dress","shirt":"shirt","jeans":"pants","shorts":"shorts",
            "hoodie":"sweater","cardigan":"sweater","suit":"jacket"}
style_items = []
for f, lab in labs.items():
    if os.path.exists(f):
        cats = [ITEM_MAP[i] for i in lab.get("item", []) if i in ITEM_MAP]
        if cats:
            style_items.append((f, cats[0]))
print(f"换装{len(TP)} 穿搭{len(style_items)}", flush=True)

CATS = ["dress","top","shirt","pants","skirt","coat","jacket","shorts","sweater"]

class RegUNetAux(nn.Module):
    def __init__(s, ch=48):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.GroupNorm(8,o), nn.SiLU(), nn.Conv2d(o,o,3,padding=1), nn.GroupNorm(8,o), nn.SiLU())
        s.d1 = blk(6, ch); s.d2 = blk(ch, ch*2); s.d3 = blk(ch*2, ch*3)
        s.mid = blk(ch*3, ch*3)
        s.u3 = blk(ch*6, ch*2); s.u2 = blk(ch*4, ch); s.u1 = blk(ch*2, ch)
        s.out = nn.Conv2d(ch, 3, 1)
        s.hd_gender = nn.Linear(ch, 2); s.hd_measure = nn.Linear(ch, 6)
        s.hd_aux_cat = nn.Linear(ch*3, len(CATS))  # v18:穿搭辅助头(挂瓶颈)
        s.pool = nn.MaxPool2d(2)
    def forward(s, inp):
        h1 = s.d1(inp); h2 = s.d2(s.pool(h1)); h3 = s.d3(s.pool(h2))
        m = s.mid(s.pool(h3))
        bottleneck_feat = m.mean((2,3))
        aux_cat = s.hd_aux_cat(bottleneck_feat)
        u3 = s.u3(torch.cat([nn.functional.interpolate(m, scale_factor=2, mode="nearest"), h3],1))
        u2 = s.u2(torch.cat([nn.functional.interpolate(u3, scale_factor=2, mode="nearest"), h2],1))
        u1 = s.u1(torch.cat([nn.functional.interpolate(u2, scale_factor=2, mode="nearest"), h1],1))
        f = u1.mean((2,3))
        return torch.tanh(s.out(u1)), s.hd_gender(f), s.hd_measure(f), aux_cat

# 数据集:混合(换装对+穿搭单图)
class MixDS(Dataset):
    def __len__(s): return len(TP)*2 + len(style_items)
    def __getitem__(s, i):
        if i < len(TP)*2:
            p = TP[i % len(TP)]
            src = Image.open(p).convert("RGB")
            W, H = src.size
            img = src.crop((0, 0, W, int(H*0.62))).resize((256,256))
            x = torch.tensor(np.asarray(img)/127.5-1, dtype=torch.float32).permute(2,0,1)
            blur = TF.gaussian_blur(x, [9,9])
            cond = TF.resize(x[:, 80:187, 93:163], (256, 256))
            inp = torch.cat([blur, cond], 0)
            ycat = torch.full((len(CATS),), -1.0)  # -1=不计损
            return inp, x, ycat
        else:
            f, cat = style_items[(i - len(TP)*2) % len(style_items)]
            img = Image.open(f).convert("RGB").resize((256,256))
            x = torch.tensor(np.asarray(img)/127.5-1, dtype=torch.float32).permute(2,0,1)
            blur = TF.gaussian_blur(x, [9,9])
            cond = TF.resize(x[:, 80:187, 93:163], (256, 256))
            inp = torch.cat([blur, cond], 0)
            ycat = torch.full((len(CATS),), -1.0)
            ycat[CATS.index(cat)] = 1.0
            return inp, x, ycat

net = RegUNetAux().cuda()
sd = torch.load(r"B:\mode\models\vyton_try_student_v17.pt", map_location="cuda", weights_only=True)
own = net.state_dict()
loaded = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
own.update(loaded)
net.load_state_dict(own)
print(f"v17迁移{len(loaded)}层", flush=True)

dl = DataLoader(MixDS(), batch_size=16, shuffle=True)
opt = torch.optim.AdamW(net.parameters(), lr=1.2e-4)
lf = nn.MSELoss()
lf_bce = nn.BCEWithLogitsLoss()
for ep in range(60):
    net.train()
    tot = tot_a = 0; nb = 0
    for inp, y, ycat in dl:
        inp, y, ycat = inp.cuda(), y.cuda(), ycat.cuda()
        out, g, m, aux = net(inp)
        loss = lf(out, y)
        mask = ycat[:, 0] > -0.5
        if mask.any():
            loss_aux = lf_bce(aux[mask], (ycat[mask] > 0).float())
            loss = loss + 0.25 * loss_aux
            tot_a += loss_aux.item()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    if (ep+1) % 20 == 0:
        torch.save(net.state_dict(), r"B:\mode\models\vyton_try_student_v18.pt")
        print(f"ep{ep+1} recon={tot/nb:.4f} aux={tot_a/max(nb,1):.4f}", flush=True)
print("V18_DONE", flush=True)