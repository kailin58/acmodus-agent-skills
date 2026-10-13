# v9达标率验证(与v8.1同口径)
import torch, glob, os
import numpy as np
from PIL import Image

src = open(r"B:/mode/glow_data/train_glow_v8.py", encoding="utf-8").read()
i0 = src.find("class GlowNet"); i1 = src.find("net = GlowNet")
ns = {}
exec("import torch\nimport torch.nn as nn\n" + src[i0:i1 if i1>i0 else i0+1500], ns)
net = ns["GlowNet"]().cuda().eval()
ck = torch.load("B:/mode/models/vyton_glow_v9.pt", map_location="cuda")
net.load_state_dict(ck["net"])
alpha = ck["alpha"]
def load(p):
    return torch.tensor(np.asarray(Image.open(p).convert("RGB").resize((192,192)))/127.5-1, dtype=torch.float32).permute(2,0,1).unsqueeze(0).cuda()
sc = []
for p in sorted(glob.glob(r"B:/mode/glow_data/pairs/*_a.jpg"))[:20]:
    mk = p.replace("_a.jpg", "_b.jpg")
    x, y = load(p), load(mk)
    with torch.no_grad():
        out = x + alpha * (net(x) - x)
    tgt = (y-x).abs().mean().item()
    got = (out-x).abs().mean().item()
    sc.append(max(0, 1 - abs(got-tgt)/max(tgt, 1e-6)))
print(f"v9达标率(alpha={alpha:.3f}): {np.mean(sc):.1%} (v8的0.7档=12.9%)")
print("V9_EVAL_DONE")
