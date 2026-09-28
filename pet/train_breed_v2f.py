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
        img = Image.open(io.BytesIO(b["bytes"])).convert("RGB").resize((224,224))
        x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1)
        if torch.rand(1).item()<0.5:
            x = torch.flip(x, dims=[2])
        # v2c强增强
        if torch.rand(1).item()<0.4:
            ang = int(torch.randint(-12, 13, (1,)).item())
            x = torch.rot90(x, k=ang//90, dims=[1,2]) if ang%90==0 else x
            import torchvision.transforms.functional as TF
            from PIL import Image as PILImage
            import numpy as _np
            arr = (x.permute(1,2,0).numpy()*255).astype("uint8")
            pil = PILImage.fromarray(arr).rotate(ang, resample=PILImage.BILINEAR)
            x = torch.tensor(_np.array(pil)/255., dtype=torch.float32).permute(2,0,1)
        if torch.rand(1).item()<0.4:
            x = x * torch.tensor([1.06, 0.96, 1.0], dtype=torch.float32).view(3,1,1)
            x = x.clamp(0,1)
        return x, y

t0 = time.time()
tr = DS(tr_files)
va = DS([te_file])
print("加载", round(time.time()-t0), "s", flush=True)

net = models.efficientnet_b3(weights="IMAGENET1K_V1")
net.classifier[1] = nn.Linear(net.classifier[1].in_features, NL)
net = net.cuda()
opt = torch.optim.AdamW(net.parameters(), lr=1.8e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6)
lf = nn.CrossEntropyLoss()
best = 0
for ep in range(6):
    net.train()
    for x, y in DataLoader(tr, batch_size=32, shuffle=True, num_workers=0):
        # mixup
        if torch.rand(1).item() < 0.5:
            lam = float(torch.distributions.Beta(0.2, 0.2).sample())
            idx = torch.randperm(len(x))
            x2 = x[idx]; y2 = y[idx]
            xm = lam*x + (1-lam)*x2
            loss = lam*lf(net(xm.cuda()), y.cuda()) + (1-lam)*lf(net(xm.cuda()), y2.cuda())
        else:
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
        torch.save(net.state_dict(), r"B:\mode\models\vyton_breed_v2f_enb3.pt")
    sch.step()
print("BREED_V2F_DONE", round(best,4), flush=True)
