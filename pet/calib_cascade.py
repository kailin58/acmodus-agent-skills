# 级联阈值校准:统计各exit置信分布+正确率vs阈值
import io, glob
import pyarrow.parquet as pq
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models

class CascadeNet(nn.Module):
    def __init__(s, ncls):
        super().__init__()
        m = models.resnet34()
        s.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        s.layer1, s.layer2, s.layer3, s.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        s.fc = nn.Linear(512, ncls)
        s.exit1 = nn.Linear(64, ncls)
        s.exit2 = nn.Linear(128, ncls)
    def forward(s, x, depth=3, ret_all=False):
        h = s.stem(x)
        h1 = s.layer1(h)
        p1 = s.exit1(h1.mean((2,3)))
        h2 = s.layer2(h1)
        p2 = s.exit2(h2.mean((2,3)))
        h3 = s.layer3(h2)
        h4 = s.layer4(h3)
        p3 = s.fc(h4.mean((2,3)))
        if ret_all: return p1, p2, p3
        return p3, 3

class DS(Dataset):
    def __init__(s, f):
        pf = pq.ParquetFile(f)
        s.items = []
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            s.items += [(x.as_py(), y.as_py()) for x, y in zip(t.column("pixel_values"), t.column("label"))]
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        b, y = s.items[i]
        img = Image.open(io.BytesIO(b["bytes"])).convert("RGB").resize((192,192))
        return torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1), y

net = CascadeNet(120).cuda()
net.load_state_dict(torch.load(r"B:\mode\models\vyton_breed_cascade.pt", map_location="cuda", weights_only=True))
net.eval()
ds = DS(r"B:\mode\pet\stanford_parquet\test-00000-of-00001.parquet")
stats = {1: [], 2: [], 3: []}
acc_at = {1: [], 2: []}
with torch.no_grad():
    for x, y in DataLoader(ds, batch_size=128):
        p1, p2, p3 = net(x.cuda(), ret_all=True)
        for pi, lvl in [(p1,1),(p2,2),(p3,3)]:
            conf = torch.softmax(pi,1).max(1).values.cpu()
            ok = (pi.argmax(1).cpu()==y)
            stats[lvl].append((conf, ok))
for lvl in [1,2,3]:
    confs = torch.cat([c for c,_ in stats[lvl]])
    oks = torch.cat([o for _,o in stats[lvl]])
    print(f"exit{lvl}: 置信分位50/75/90 = {confs.quantile(0.5):.3f}/{confs.quantile(0.75):.3f}/{confs.quantile(0.9):.3f} | 全集acc={oks.float().mean():.4f}", flush=True)
    # 各阈值下的覆盖率和精度
    for th in [0.6, 0.7, 0.75, 0.8, 0.85]:
        m = confs > th
        if m.sum() > 0:
            print(f"  th={th}: 覆盖{m.float().mean():.2f} 精度={oks[m].float().mean():.4f}", flush=True)
