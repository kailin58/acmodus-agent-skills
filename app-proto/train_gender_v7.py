
import io
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
FILES = [r"B:\mode\face_data\train-00000-of-00004-e715178553977907.parquet",
         r"B:\mode\face_data\train-00001-of-00004-f38b58e3987f3fbf.parquet",
         r"B:\mode\face_data\train-00002-of-00004-239e931aa9c3b3e6.parquet",
         r"B:\mode\face_data\train-00003-of-00004-847c279691a19548.parquet"]
dfs = [pd.read_parquet(f, columns=["image","gender"]) for f in FILES]
df = pd.concat(dfs, ignore_index=True)
print("全量:", len(df), flush=True)
class DS(Dataset):
    def __init__(self, rows):
        self.rows = [(r["image"], int(r["gender"])) for _, r in rows.iterrows()]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        b, g = self.rows[i]
        img = Image.open(io.BytesIO(b.get("bytes"))).convert("RGB").resize((112,112))
        import random
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.array(img, dtype=np.float32)/255.0
        if random.random() < 0.5:
            arr = arr * (0.8 + random.random()*0.4)
        if random.random() < 0.3:  # 色彩抖动
            arr = arr * np.array([1+random.uniform(-0.08,0.08), 1+random.uniform(-0.08,0.08), 1+random.uniform(-0.08,0.08)], np.float32)
        return torch.tensor(np.clip(arr,0,1)).permute(2,0,1), g
class GenderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,48,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48,96,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(96,192,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(192,96); self.head = nn.Linear(96,2)
    def forward(self,x):
        return self.head(torch.relu(self.fc(self.net(x).flatten(1))))
n = len(df); split = int(n*0.92)
trs = DS(df.iloc[:split]); vas = DS(df.iloc[split:])
m = GenderNet().cuda()
gs = df.iloc[:split]["gender"].value_counts().to_dict()
w = torch.tensor([1.0, gs.get(0,1)/max(gs.get(1,1),1)]).cuda()
lf = nn.CrossEntropyLoss(weight=w)
opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=22)
best = 0
for ep in range(22):
    m.train()
    for x, g in DataLoader(trs, batch_size=96, shuffle=True):
        x, g = x.cuda(), g.cuda()
        loss = lf(m(x), g)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); corr = tot = 0
    per = {0:[0,0],1:[0,0]}
    with torch.no_grad():
        for x, g in DataLoader(vas, batch_size=128):
            p = m(x.cuda()).argmax(1).cpu()
            corr += (p==g).sum().item(); tot += len(g)
            for pi, gi in zip(p.tolist(), g.tolist()):
                per[gi][0 if pi==gi else 1] += 1
    acc = corr/tot
    print(f"ep{ep+1} acc={acc:.4f} M={per[0][0]/max(sum(per[0]),1):.3f} F={per[1][0]/max(sum(per[1]),1):.3f}", flush=True)
    if acc > best:
        best = acc
        torch.save(m.state_dict(), r"B:\mode\models\vyton_gender_face_v7.pt")
sch.step()
print("GENDER_V7_DONE", round(best,4), flush=True)
