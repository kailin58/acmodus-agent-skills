# 品种级联早退版:主干+早退头1(1/3处)+早退头2(2/3处)+终头
# 推理:浅头置信>0.92直接返回,否则逐级深入
import io, glob, time
import pyarrow.parquet as pq
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models

tr_files = sorted(glob.glob(r"B:\mode\pet\stanford_parquet\train-*.parquet"))
te_file = r"B:\mode\pet\stanford_parquet\test-00000-of-00001.parquet"

class DS(Dataset):
    def __init__(s, files):
        s.items = []
        for f in files:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg)
                s.items += [(x.as_py(), y.as_py()) for x, y in zip(t.column("pixel_values"), t.column("label"))]
        print("样本:", len(s.items), flush=True)
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        b, y = s.items[i]
        img = Image.open(io.BytesIO(b["bytes"])).convert("RGB").resize((192,192))
        x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1)
        if torch.rand(1).item()<0.5: x = torch.flip(x, dims=[2])
        return x, y

class CascadeNet(nn.Module):
    """resnet34主干+3个早退头"""
    def __init__(s, ncls):
        super().__init__()
        m = models.resnet34(weights="IMAGENET1K_V1")
        s.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        s.layer1, s.layer2, s.layer3, s.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        s.fc = nn.Linear(512, ncls)
        s.exit1 = nn.Linear(64, ncls)    # layer1后(浅,1/8算力)
        s.exit2 = nn.Linear(128, ncls)   # layer2后(中,1/3算力)
    def forward(s, x, depth=3):
        h = s.stem(x)
        h1 = s.layer1(h)
        if depth >= 1:
            p1 = s.exit1(h1.mean((2,3)))
            c1 = torch.softmax(p1, 1).max(1).values
            if depth == 3 and (c1 > 0.92).all():
                return p1, 1
        h2 = s.layer2(h1)
        if depth >= 2:
            p2 = s.exit2(h2.mean((2,3)))
            c2 = torch.softmax(p2, 1).max(1).values
            if depth == 3 and (c2 > 0.95).all():
                return p2, 2
        h3 = s.layer3(h2)
        h4 = s.layer4(h3)
        return s.fc(h4.mean((2,3))), 3

NL = 120
net = CascadeNet(NL).cuda()
tr = DS(tr_files); va = DS([te_file])
# 从v2e最佳权重初始化主干(resnet34部分)
sd = torch.load(r"B:\mode\models\vyton_breed_v2e_enb3.pt", map_location="cuda", weights_only=True)
# v2e是enb3——结构不同,只load兼容的stem/layer名(失败则从头微调)
try:
    own = net.state_dict()
    loaded = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    own.update(loaded)
    net.load_state_dict(own)
    print("迁移:", len(loaded), "层", flush=True)
except Exception as e:
    print("迁移跳过:", e, flush=True)

opt = torch.optim.AdamW(net.parameters(), lr=1.5e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
lf = nn.CrossEntropyLoss()
for ep in range(4):
    net.train()
    for x, y in DataLoader(tr, batch_size=48, shuffle=True):
        p3, d = net(x.cuda())
        loss = lf(p3, y.cuda())
        # 辅助监督:强推depth全走(训练时depth=3逐级softmax不过早退)
        opt.zero_grad(); loss.backward(); opt.step()
    # eval:精度+早退率
    net.eval(); c=t=0; ex={1:0, 2:0, 3:0}
    with torch.no_grad():
        for x, y in DataLoader(va, batch_size=64):
            p, d = net(x.cuda())
            c += (p.argmax(1).cpu()==y).sum().item(); t += len(y)
            ex[d] += len(y)
    print(f"ep{ep+1} acc={c/t:.4f} exits={ex}", flush=True)
    torch.save(net.state_dict(), r"B:\mode\models\vyton_breed_cascade.pt")
    sch.step()
print("CASCADE_DONE", flush=True)
