# 品种v2:Stanford 120犬种 resnet34微调(parquet流式)
import io, os, glob, time
import pyarrow.parquet as pq
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models

tr_files = sorted(glob.glob(r"B:\mode\pet\stanford_parquet\train-*.parquet"))
te_file = r"B:\mode\pet\stanford_parquet\test-00000-of-00001.parquet"
# 先扫label范围
labels = set()
rows_tr = []
for f in tr_files:
    pf = pq.ParquetFile(f)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["label"])
        labels.update(x.as_py() for x in t.column("label"))
NL = max(labels)+1
print("类别数:", NL, flush=True)

class DS(Dataset):
    def __init__(s, files, train=True):
        s.items = []
        for f in files:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg)
                labs = [x.as_py() for x in t.column("label")]
                imgs = [x.as_py() for x in t.column("pixel_values")]
                s.items += list(zip(imgs, labs))
        print("样本:", len(s.items), flush=True)
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        b, y = s.items[i]
        img = Image.open(io.BytesIO(b["bytes"])).convert("RGB").resize((160,160))
        x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1)
        if torch.rand(1).item()<0.5:
            x = torch.flip(x, dims=[2])
        return x, y

t0 = time.time()
tr = DS(tr_files)
va = DS([te_file])
print("加载", round(time.time()-t0), "s", flush=True)

net = models.resnet34()
net.fc = nn.Linear(net.fc.in_features, NL)
net.load_state_dict(torch.load(r"B:\mode\models\vyton_breed_v2_120.pt", map_location="cuda", weights_only=True))
net.fc = nn.Linear(net.fc.in_features, NL)
net = net.cuda()
opt = torch.optim.AdamW(net.parameters(), lr=8e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
lf = nn.CrossEntropyLoss()
best = 0
for ep in range(4):
    net.train()
    for x, y in DataLoader(tr, batch_size=48, shuffle=True, num_workers=0):
        loss = lf(net(x.cuda()), y.cuda())
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); c=t=0
    with torch.no_grad():
        for x, y in DataLoader(va, batch_size=64):
            c += (net(x.cuda()).argmax(1).cpu()==y).sum().item(); t += len(y)
    acc = c/t
    print(f"ep{ep+1} acc={acc:.4f}", flush=True)
    if acc > best:
        best = acc
        torch.save(net.state_dict(), r"B:\mode\models\vyton_breed_v2b_120.pt")
    sch.step()
print("BREED_V2B_DONE", round(best,4), flush=True)
