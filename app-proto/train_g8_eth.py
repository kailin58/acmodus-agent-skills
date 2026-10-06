# 性别v8:FairFace val自蒸馏+民族图伪标签(公平性)
import io, os, glob, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

fs = glob.glob(r"B:\mode\face_data\*.parquet")
valf = [f for f in fs if "val" in f.lower()][0]
df = pd.read_parquet(valf, columns=["image","gender"])
print("fairface:", len(df), flush=True)

class GenderNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(3,48,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
                              nn.Conv2d(48,96,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
                              nn.Conv2d(96,192,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(192,96); s.head = nn.Linear(96,2)
    def forward(s,x): return s.head(torch.relu(s.fc(s.net(x).flatten(1))))

m = GenderNet().cuda()
m.load_state_dict(torch.load(r"B:\mode\models\vyton_gender_face_v5.pt", map_location="cuda", weights_only=True))
m.eval()
pseudo = []
with torch.no_grad():
    for d in glob.glob(r"B:\mode\ethnic_data\*"):
        for f in glob.glob(os.path.join(d, "*.jpg"))[:3]:
            img = Image.open(f).convert("RGB").resize((112,112))
            x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).cuda()
            conf = torch.softmax(m(x),1).max().item()
            if conf > 0.9:
                pseudo.append((f, m(x).argmax(1).item()))
print("民族伪标:", len(pseudo), flush=True)

class DS(Dataset):
    def __len__(s): return len(s.items)
    def __init__(s, items): s.items = items
    def __getitem__(s, i):
        b, gn = s.items[i]
        if isinstance(b, dict): b = b["bytes"]
        elif isinstance(b, str):
            img = Image.open(b).convert("RGB").resize((112,112))
            x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1)
            if torch.rand(1).item()<0.5: x = torch.flip(x, dims=[2])
            return x, gn
        img = Image.open(io.BytesIO(b)).convert("RGB").resize((112,112))
        x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1)
        if torch.rand(1).item()<0.5: x = torch.flip(x, dims=[2])
        return x, gn

items = [(r["image"], int(r["gender"])) for _, r in df.iterrows()] + pseudo
random.seed(7); random.shuffle(items)
cut = len(items)*9//10
tr, va = DS(items[:cut]), DS(items[cut:])
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=3e-5)
lf = nn.CrossEntropyLoss()
for ep in range(3):
    for x, g_ in DataLoader(tr, batch_size=128):
        loss = lf(m(x.cuda()), g_.cuda())
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); c=t=0
    with torch.no_grad():
        for x, g_ in DataLoader(va, batch_size=128):
            c += (m(x.cuda()).argmax(1).cpu()==g_).sum().item(); t += len(g_)
    print(f"ep{ep+1} acc={c/t:.4f}", flush=True)
    torch.save(m.state_dict(), r"B:\mode\models\vyton_gender_face_v8_eth.pt")
    m.train()
print("G8_DONE")
