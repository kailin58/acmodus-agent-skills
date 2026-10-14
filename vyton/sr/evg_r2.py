# R2视觉验收:三联(ETC1S裸|R2增强|原图)
import torch, torch.nn as nn, glob
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, r"B:/mode/vyton/sr")
exec(open(r"B:/mode/vyton/sr/train_r2.py", encoding="utf-8").read().split("BASE =")[0])
net = RebuildNet().cuda().eval()
net.load_state_dict(torch.load(r"B:/mode/models/sr/rebuild_head.pt", map_location="cuda"))
rows = []
for k in [3, 17, 42, 88, 150, 233]:
    lo_p = rf"B:/mode/models/sr/pairs/lo/{k:04d}.png"
    hi_p = rf"B:/mode/models/sr/pairs/hi/{k:04d}.png"
    try:
        lo = np.asarray(Image.open(lo_p).convert("RGB").resize((256,256)))/127.5-1
        hi = np.asarray(Image.open(hi_p).convert("RGB").resize((256,256)))
    except Exception:
        continue
    x = torch.tensor(lo, dtype=torch.float32).permute(2,0,1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = net(x)[0].cpu().numpy().transpose(1,2,0)
    out = ((out+1)*127.5).clip(0,255).astype(np.uint8)
    row = np.concatenate([np.asarray(Image.open(lo_p).convert("RGB").resize((256,256))), out, hi], axis=1)
    rows.append(row)
grid = np.concatenate(rows, axis=0)
Image.fromarray(grid).save(r"B:/mode/models/sr/r2_evidence.png")
print("R2_EVG_OK", len(rows))
