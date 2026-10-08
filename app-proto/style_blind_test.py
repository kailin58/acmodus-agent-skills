import glob, json, random, os
import numpy as np
from PIL import Image, ImageDraw
import torch, torch.nn as nn
import torchvision.models as models

STYLES = ["minimalist","french","vintage","street","sporty","elegant","casual","boho"]
SCENES = ["work","date","party","casual","travel","home"]

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        m = models.efficientnet_b3(weights=None)
        s.back = m
        f = m.classifier[1].in_features
        s.hd_cat = nn.Linear(f, 9); s.hd_col = nn.Linear(f, 12)
        s.hd_sty = nn.Linear(f, 8); s.hd_sce = nn.Linear(f, 6)
    def forward(s, x):
        h = s.back.features(x)
        h = nn.functional.adaptive_avg_pool2d(h, 1).flatten(1)
        return s.hd_cat(h), s.hd_col(h), s.hd_sty(h), s.hd_sce(h)

net = Net().cuda()
sd = torch.load(r"B:\mode\models\vyton_style_v9_multi.pt", map_location="cuda", weights_only=True)
net.load_state_dict(sd); net.eval()

labs = json.load(open(r"B:\mode\style_train\pin_labels_v1.json", encoding="utf-8"))
files = sorted(labs.keys())
va_files = [f for f in files[:max(1, len(files)//8)] if os.path.exists(f)]
random.seed(7)
picks = random.sample(va_files, min(8, len(va_files)))

def top3(logits, names):
    p = torch.sigmoid(logits[0])
    idx = p.argsort(descending=True)[:3]
    return [(names[i], round(p[i].item(), 2)) for i in idx]

out = []
for i, f in enumerate(picks):
    img = Image.open(f).convert("RGB")
    x = torch.tensor(np.array(img.resize((160,160)))/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).cuda()
    with torch.no_grad():
        pc, pl, ps, pe = net(x)
    st = top3(ps, STYLES); sc = top3(pe, SCENES)
    tk = labs[f]
    true = (tk.get("style") or [])[:2] + (tk.get("scene") or [])[:2]
    out.append((f, st, sc, true))

# 拼证据图:4联(每格=原图+真值词+模型判定)
grid = Image.new("RGB", (1600, 800), (15,15,20))
dr = ImageDraw.Draw(grid)
for i, (f, st, sc, true) in enumerate(out[:4]):
    im = Image.open(f).convert("RGB").resize((380, 380))
    x0, y0 = (i%2)*800+10, (i//2)*400+10
    grid.paste(im, (x0, y0))
    tx = x0+400
    dr.text((tx, y0+20), f"TRUE: {true}", fill=(120,220,120))
    dr.text((tx, y0+60), f"style: {st[0][0]} {st[0][1]} | {st[1][0]} {st[1][1]}", fill=(240,240,120))
    dr.text((tx, y0+100), f"scene: {sc[0][0]} {sc[0][1]} | {sc[1][0]} {sc[1][1]}", fill=(120,180,240))
grid.save(r"B:\mode\style_train\v9_blind_evidence.png")
print("EVIDENCE_DONE")
for i, (f, st, sc, true) in enumerate(out):
    print(f"[{i}] true={true} | model: sty={st[0][0]}({st[0][1]}) sce={sc[0][0]}({sc[0][1]})")