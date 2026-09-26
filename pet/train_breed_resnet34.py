# Vyton-Pet品种v1.1:resnet34预训练微调
import pandas as pd, io
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models, transforms

df = pd.read_parquet(r"B:\mode\pet\breed_data\train.parquet")
tf_tr = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
tf_va = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
class DS(Dataset):
    def __init__(self, rows, tf):
        self.items = [(r["image"]["bytes"], int(r["label"])) for _, r in rows.iterrows()]
        self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        b, y = self.items[i]
        img = Image.open(io.BytesIO(b)).convert("RGB").resize((224,224))
        return self.tf(img), y

n = len(df); sp = int(n*0.9)
tr = DS(df.iloc[:sp].iterrows().__class__ and df.iloc[:sp], tf_tr)
va = DS(df.iloc[sp:], tf_va)
m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
m.fc = nn.Linear(m.fc.in_features, 37)
m = m.cuda()
# 分层lr:底座小/头大
opt = torch.optim.AdamW([{"params": list(m.children())[:-1][0].parameters(), "lr": 2e-5}] *
                        1 + [{"params": m.fc.parameters(), "lr": 1e-3}]) if False else torch.optim.AdamW([
    {"params": [p for n_,p in m.named_parameters() if not n_.startswith("fc")], "lr": 2e-5},
    {"params": m.fc.parameters(), "lr": 1e-3}])
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
lf = nn.CrossEntropyLoss()
best = 0
for ep in range(12):
    m.train()
    for x, y in DataLoader(tr, batch_size=48, shuffle=True):
        loss = lf(m(x.cuda()), y.cuda())
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); c=t=0
    with torch.no_grad():
        for x, y in DataLoader(va, batch_size=64):
            c += (m(x.cuda()).argmax(1).cpu()==y).sum().item(); t += len(y)
    acc = c/t
    print(f"ep{ep+1} acc={acc:.4f}", flush=True)
    if acc > best:
        best = acc
        torch.save(m.state_dict(), r"B:\mode\models\vyton_pet_breed_resnet34.pt")
    sch.step()
print("BREED_RN_DONE", round(best,4), flush=True)
