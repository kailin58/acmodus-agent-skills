# 虚拟自己v15:身份条件头驱动点彩(ID embedding→个性化调色板)
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageFilter

class IDNet(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,2,1), nn.BatchNorm2d(o), nn.ReLU())
        s.net = nn.Sequential(blk(3,32), blk(32,64), blk(64,128), blk(128,256), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(256, 128)
    def forward(s, x): return F.normalize(s.fc(s.net(x).flatten(1)), dim=1)

net = IDNet().cuda()
net.load_state_dict(torch.load(r"B:\mode\models\vyton_id_head.pt", map_location="cuda", weights_only=True))
net.eval()

face_img = Image.open(r"B:\mode\vton_training\face_test.png").convert("RGB")
x = torch.tensor(np.asarray(face_img.resize((112,112)))/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).cuda()
with torch.no_grad():
    emb = net(x)[0].cpu().numpy()
print("身份向量维度:", emb.shape, "前8维:", emb[:8].round(3))

# ID向量→确定性个人调色板(身份哈希种子)
seed = int(np.abs(emb).sum() * 10000) % (2**31)
rng = np.random.default_rng(seed)
# 基于ID的个性化:发色/肤色微调(保持v9红棕基底,亮度/饱和按ID微调)
id_bright = 0.9 + float(emb[:16].mean()) * 0.15   # ID亮度签名
id_sat = 0.95 + float(emb[16:32].std()) * 0.1     # ID饱和签名
print(f"ID签名: 亮度×{id_bright:.3f} 饱和×{id_sat:.3f} seed={seed}")

# 跑v9基底+ID调制
import subprocess, sys
s9 = open(r"B:\mode\pinterest\acmodus-agent-skills\3d-tryon\f6_v9_strands.py", encoding="utf-8").read()
# 注入:红棕主层色值乘ID签名
s15 = s9.replace("if u_ < 0.70:      # 70%红棕主层(v12加量)\n        r_, g_, b_ = int(100+rng8.uniform(0,55)), int(56+rng8.uniform(0,28)), int(30+rng8.uniform(0,18))",
                 f"if u_ < 0.70:\n        r_, g_, b_ = int((100+rng8.uniform(0,55))*{id_bright:.3f}), int((56+rng8.uniform(0,28))*{id_sat:.3f}), int((30+rng8.uniform(0,18))*{id_bright:.3f})") if "v12加量" in s9 else s9
# 兜底:直接按v9真实字符串处理
import re
s15 = re.sub(r"r_, g_, b_ = int\(10[05]\+rng8\.uniform\(0,55\)\), int\(5[68]\+rng8\.uniform\(0,2[68]\)\), int\([23][08]\+rng8\.uniform\(0,1[68]\)\)",
             lambda m: f"r_, g_, b_ = int((100+rng8.uniform(0,55))*{id_bright:.3f}), int((56+rng8.uniform(0,28))*{id_sat:.3f}), int((30+rng8.uniform(0,18))*{id_bright:.3f})", s15)
s15 = s15.replace("VIRTUAL SELF v9 (auburn strands)", "VIRTUAL SELF v15 (ID-conditioned)")
s15 = s15.replace("virtual_self_v9.png", "virtual_self_v15.png")
s15 = s15.replace("V9_DONE", "V15_DONE")
open(r"B:\mode\pinterest\acmodus-agent-skills\3d-tryon\f6_v15_id.py", "w", encoding="utf-8").write(s15)
print("v15脚本OK(ID签名注入)")
