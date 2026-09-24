
# Vyton-Style v2:扩数据(150新素材)+多标签分类+温度校准头
import os, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMG_DIRS = [r"B:\mode\vton_full\raw", r"B:\mode\pinterest\images"]
OLD = json.load(open(r"B:\mode\style_train\weak_labels.json", encoding="utf-8"))
COLORS = ["白","黑","红","蓝","绿","粉","灰","米","棕","黄","紫","橙"]
CATS = ["上衣","T恤","裤","半裙","短裤","裙","针织","外套","大衣"]
device = "cuda"

# 新素材弱标签(用颜色规则v1引擎自动标注颜色)
import colorsys
from collections import Counter
def img_colors(path, k=2):
    img = Image.open(path).convert("RGB").resize((96,128))
    a = np.array(img).reshape(-1,3)
    def n2c(r,g,b):
        v=(int(r)+int(g)+int(b))/3; mx,mn=max(r,g,b),min(r,g,b)
        if mx-mn<25:
            return None
        h,s,vv=colorsys.rgb_to_hsv(r/255,g/255,b/255)
        if s<0.2: return None
        hd=h*360
        return "红" if (hd<15 or hd>=345) else "橙" if hd<40 else "黄" if hd<68 else "绿" if hd<160 else "蓝" if hd<200 else "紫" if hd<300 else "粉"
    cnt = Counter(n2c(*px) for px in a[::11])
    cnt.pop(None, None)
    tot = sum(cnt.values())+1
    return [c for c,n in cnt.most_common(4) if n/tot>0.12][:k]

data = list(OLD)
new_dir = r"B:\mode\pinterest\images"
added = 0
for f in os.listdir(new_dir):
    if not f.lower().endswith((".jpg",".png",".webp")): continue
    try:
        cs = img_colors(os.path.join(new_dir, f))
        if cs:
            data.append({"file": os.path.join(new_dir,f), "img": f, "colors": cs, "cats": []})
            added += 1
    except Exception: pass
print("旧数据:", len(OLD), "新增:", added, flush=True)

class DS(Dataset):
    def __init__(self, data):
        self.data = [d for d in data if d["colors"] or d["cats"]]
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]
        fp = d.get("file") or os.path.join(r"B:\mode\vton_full\raw", d.get("img",""))
        img = Image.open(fp).convert("RGB").resize((160,160))
        x = torch.tensor(np.array(img)/255.0, dtype=torch.float32).permute(2,0,1)
        yc = torch.zeros(len(COLORS)); yk = torch.zeros(len(CATS))
        for c in d["colors"]:
            if c in COLORS: yc[COLORS.index(c)] = 1
        for c in d["cats"]:
            if c in CATS: yk[CATS.index(c)] = 1
        return x, yc, yk

class StyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.back = nn.Sequential(
            nn.Conv2d(3,24,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,2,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48,96,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(96,128)
        self.head_c = nn.Linear(128,len(COLORS))
        self.head_k = nn.Linear(128,len(CATS))
        self.logT = nn.Parameter(torch.zeros(1))  # 温度校准(RLCD方向)
    def forward(self,x):
        f = self.back(x).flatten(1)
        f = torch.relu(self.fc(f))
        return self.head_c(f)/self.logT.exp(), self.head_k(f)/self.logT.exp()

ds = DS(data)
n = len(ds); tr, va = int(n*0.9), n-int(n*0.9)
trs, vas = torch.utils.data.random_split(ds, [tr,va])
m = StyleNet().to(device)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
# 类别失衡:正样本加权
pc = torch.zeros(len(COLORS)).to(device)
for d in data:
    for c in d["colors"]:
        if c in COLORS: pc[COLORS.index(c)] += 1
pos_w = (len(data) - pc) / pc.clamp(min=1)
pk = torch.zeros(len(CATS)).to(device)
for d in data:
    for c in d["cats"]:
        if c in CATS: pk[CATS.index(c)] += 1
pos_wk = (len(data) - pk) / pk.clamp(min=1)
lfc = nn.BCEWithLogitsLoss(pos_weight=pos_w)
lfk = nn.BCEWithLogitsLoss(pos_weight=pos_wk)
t0 = time.time()
for ep in range(12):
    m.train()
    for x, yc, yk in DataLoader(trs, batch_size=24, shuffle=True):
        x,yc,yk = x.to(device),yc.to(device),yk.to(device)
        lc,lk = m(x)
        loss = lfc(lc,yc)+0.5*lfk(lk,yk)
        opt.zero_grad(); loss.backward(); opt.step()
    # 验证+ECE(校准误差)
    m.eval(); hits=tot=0; confs=[]; corrects=[]
    with torch.no_grad():
        for x, yc, yk in DataLoader(vas, batch_size=32):
            x=x.to(device)
            lc,lk = m(x)
            p = torch.sigmoid(lc).cpu()
            pred = (p>0.5).float()
            hits += (pred*yc).sum().item(); tot += yc.numel()
            confs.append(p.max(1).values); corrects.append((pred*yc).max(1).values)
    conf = torch.cat(confs); corr = torch.cat(corrects)
    ece = (conf-corr).abs().mean().item()  # 简化ECE
    print(f"ep{ep+1} loss={loss.item():.3f}", flush=True)
torch.save(m.state_dict(), r"B:\mode\models\vyton_style_v2.pt")
# 导出校准后置信度示例
print("V2_SAVED", int(time.time()-t0), "s", flush=True)
