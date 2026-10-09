# v13第一步:自评环——v12对训练数据自评,找标签可疑样本
import glob, json, os
import numpy as np
import torch, torch.nn as nn
import torchvision.models as models
from PIL import Image

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
sd = torch.load(r"B:\mode\models\vyton_style_v12_multi.pt", map_location="cuda", weights_only=True)
net.load_state_dict(sd); net.eval()
print("v12加载OK", flush=True)

labs = json.load(open(r"B:\mode\style_train\pin_labels_v1.json", encoding="utf-8"))
S_MAP = {"minimalist":"minimalist","scandinavian":"minimalist","normcore":"minimalist","french":"elegant","italian":"elegant",
         "vintage":"vintage","american vintage":"vintage","y2k":"vintage","streetwear":"street","street":"street","techwear":"street",
         "japanese":"street","korean":"street","athleisure":"sporty","sporty":"sporty","gorpcore":"sporty","boho":"boho",
         "cottagecore":"boho","smart casual":"casual","business casual":"casual","casual":"casual","goth":"street","preppy":"elegant"}
C_MAP = {"work":"work","office":"work","interview":"work","business":"work","date":"date","dinner":"date","brunch":"date",
         "party":"party","concert":"party","wedding":"party","casual":"casual","weekend":"casual","school":"casual",
         "travel":"travel","holiday":"travel","home":"home"}

suspect = []
files = [f for f in sorted(labs.keys()) if os.path.exists(f)][:800]
for f in files:
    try:
        img = Image.open(f).convert("RGB").resize((160,160))
        x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).cuda()
        with torch.no_grad():
            pc, pl, ps, pe = net(x)
        # 模型预测
        sty_p = torch.sigmoid(ps[0])
        sce_p = torch.sigmoid(pe[0])
        pred_sty = STYLES[sty_p.argmax().item()]
        pred_sce = SCENES[sce_p.argmax().item()]
        # 真值
        true_sty = [S_MAP[s] for s in labs[f].get("style", []) if s in S_MAP]
        true_sce = [C_MAP[c] for c in labs[f].get("scene", []) if c in C_MAP]
        # 冲突判定:预测高置信但与全部真值不符
        sty_conf = sty_p.max().item()
        if sty_conf > 0.5 and true_sty and pred_sty not in true_sty:
            suspect.append({"f": f, "type": "style", "true": true_sty, "pred": pred_sty, "conf": round(sty_conf,2)})
        elif sty_conf < 0.15 and not true_sce:  # 双低信息样本
            suspect.append({"f": f, "type": "lowinfo", "true": true_sty, "pred": pred_sty, "conf": round(sty_conf,2)})
    except Exception:
        pass
print(f"可疑样本: {len(suspect)}/{len(files)}", flush=True)
json.dump(suspect, open(r"B:\mode\style_train\suspect_v12.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
from collections import Counter
print("类型分布:", Counter(s["type"] for s in suspect))
print("SELF_EVAL_DONE", flush=True)