# 蒸馏:resnet34教师→MobileNetV3-Small学生(品种37类)
import pandas as pd, io
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models, transforms

df = pd.read_parquet(r"B:\mode\pet\breed_data\train.parquet")
tf = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
tf_va = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
class DS(Dataset):
    def __init__(self, rows, tf):
        self.items = [(r["image"]["bytes"], int(r["label"])) for _, r in rows.iterrows()]
        self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        b, y = self.items[i]
        return self.tf(Image.open(io.BytesIO(b)).convert("RGB").resize((224,224))), y

n = len(df); sp = int(n*0.9)
tr = DS(df.iloc[:sp], tf); va = DS(df.iloc[sp:], tf_va)

# 教师(resnet34已训)
T = models.resnet34()
T.fc = nn.Linear(T.fc.in_features, 37)
T.load_state_dict(torch.load(r"B:\mode\models\vyton_pet_breed_resnet34.pt", map_location="cuda", weights_only=True))
T = T.cuda().eval()
for p in T.parameters(): p.requires_grad_(False)

# 学生(MobileNetV3-Small ~2.5M参数≈10MB)
S = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
S.classifier[3] = nn.Linear(S.classifier[3].in_features, 37)
S = S.cuda()

opt = torch.optim.AdamW(S.parameters(), lr=2e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
ce = nn.CrossEntropyLoss()
kl = nn.KLDivLoss(reduction="batchmean")
best = 0
for ep in range(15):
    S.train()
    for x, y in DataLoader(tr, batch_size=48, shuffle=True):
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            t_logits = T(x)
        s_logits = S(x)
        # 蒸馏损失 = 0.5*硬标签CE + 0.5*软标签KL(T=6)
        loss = 0.5*ce(s_logits, y) + 0.5*kl(torch.log_softmax(s_logits/6, 1), torch.softmax(t_logits/6, 1))*36
        opt.zero_grad(); loss.backward(); opt.step()
    S.eval(); c=t=0
    with torch.no_grad():
        for x, y in DataLoader(va, batch_size=64):
            c += (S(x.cuda()).argmax(1).cpu()==y).sum().item(); t += len(y)
    acc = c/t
    print(f"ep{ep+1} acc={acc:.4f}", flush=True)
    if acc > best:
        best = acc
        torch.save(S.state_dict(), r"B:\mode\models\vyton_pet_breed_mobile_l.pt")
    sch.step()
import os
sz = os.path.getsize(r"B:\mode\models\vyton_pet_breed_mobile_l.pt")/1024/1024
print(f"DISTILL_DONE {best:.4f} size={sz:.1f}MB", flush=True)
