# 性别v4快速eval
import io
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

df = pd.read_parquet(r"B:\mode\face_data\train.parquet") if False else None
# 用fairface val parquet(1.1万)
import glob
fs = glob.glob(r"B:\mode\face_data\*.parquet")
valf = [f for f in fs if "val" in f.lower()] or fs
df = pd.read_parquet(valf[0], columns=["image","gender"])
print("val:", len(df))
class DS(Dataset):
    def __init__(s, rows): s.rows = [(r["image"], int(r["gender"])) for _, r in rows.iterrows()]
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        b, g = s.rows[i]
        img = Image.open(io.BytesIO(b["bytes"])).convert("RGB").resize((112,112))
        return torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1), g
class GenderNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(3,48,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48,96,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(96,192,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(192,96); s.head = nn.Linear(96,2)
    def forward(s,x): return s.head(torch.relu(s.fc(s.net(x).flatten(1))))
m = GenderNet().cuda()
m.load_state_dict(torch.load(r"B:\mode\models\vyton_gender_face_v5.pt", map_location="cuda", weights_only=True))
m.eval()
c=t=0
per={0:[0,0],1:[0,0]}
with torch.no_grad():
    for x, g in DataLoader(DS(df), batch_size=128):
        p = m(x.cuda()).argmax(1).cpu()
        c += (p==g).sum().item(); t += len(g)
        for pi, gi in zip(p.tolist(), g.tolist()):
            per[gi][0 if pi==gi else 1]+=1
print(f"acc={c/t:.4f} M={per[0][0]/max(sum(per[0]),1):.3f} F={per[1][0]/max(sum(per[1]),1):.3f}")
