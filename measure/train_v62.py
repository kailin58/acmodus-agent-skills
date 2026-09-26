# Vyton-Measure v6.2:合成→真实域适配(148真图fine-tune)
# MediaPipe 33点→解剖17点 + rembg轮廓→围度带
import json, os
import numpy as np
import torch, torch.nn as nn
from PIL import Image

# 1) 真实数据:landmarks_full.json(148,kps=MediaPipe 33×3)
raw = json.load(open(r"B:\mode\measure\landmarks_full.json", encoding="utf-8"))
print("真图:", len(raw), flush=True)

# MediaPipe33→解剖17映射(POSE landmarks索引)
MP2ANAT = {
 0:(0.955,1.0),   # 头顶近似(nose上方)
 2:None,5:None,   # 眼(略)
 11:(0.80,0.845),12:(0.80,0.845),  # 肩
 13:(0.62,0.67),14:(0.62,0.67),    # 肘
 15:(0.50,0.55),16:(0.50,0.55),    # 腕
 23:(0.30,0.36),24:(0.30,0.36),    # 髋
 25:(0.13,0.18),26:(0.13,0.18),    # 膝
 27:(0.01,0.05),28:(0.01,0.05),    # 踝
}
ANAT_Z = [0.98,0.93,0.93,0.82,0.82,0.65,0.65,0.52,0.52,0.55,0.43,0.16,0.16,0.03,0.03,0.33,0.33]
def mp_to_anat(kps33):
    P = kps33[:,:2]
    anat = np.zeros((17,3), np.float32)
    mp_idx = [0,2,5,11,12,13,14,15,16,23,24,25,26,27,28,23,24]
    for i, j in enumerate(mp_idx):
        anat[i,:2] = P[j]
        anat[i,2] = ANAT_Z[i]
    return anat

def width_bands(anat):
    # 真图围度带:从关键点距离近似(肩宽/髋宽/躯干长/腿长等8条)
    sh = np.linalg.norm(anat[3]-anat[4])       # 肩宽
    hip = np.linalg.norm(anat[14]-anat[15]) if len(anat)>15 else sh*0.9
    return np.array([sh*0.45, sh, sh*0.82, hip, hip*0.6, hip*0.6, sh*0.22, sh*0.22], np.float32)

feats, labels = [], []
for d in raw:
    kps = np.array(d["kps"], dtype=np.float32).reshape(33,3)
    anat = mp_to_anat(kps)
    wd = width_bands(anat)
    x = np.concatenate([anat.reshape(-1), wd])
    feats.append(x)
    labels.append([d.get("height",0.5), d.get("weight",0.5), d.get("gender",0.5),
                   d.get("muscle",0.4), d.get("age",0.4), d.get("proportions",0.5)])
X = np.array(feats, np.float32)
Y = np.array(labels, np.float32)
print("特征:", X.shape, flush=True)

# 2) 域适配fine-tune:合成v6.1权重→真图低lr微调
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(59, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 6))
    def forward(s, x): return s.net(x)

# 合成数据重算归一化统计
data = json.load(open(r"B:\mode\measure\syn_kps_v2\data.json", encoding="utf-8"))
Xs = np.array([np.array(d["kp"], dtype=np.float32).reshape(-1).tolist()+np.array(d["wd"], dtype=np.float32).tolist() for d in data], np.float32)
MU, SD = Xs.mean(0), Xs.std(0)+1e-6
Xn = (X - MU) / SD  # 真图用合成统计归一(域对齐)

m = Net().cuda()
m.load_state_dict(torch.load(r"B:\mode\models\vyton_measure_v61.pt", map_location="cuda", weights_only=True))
xt = torch.tensor(Xn).cuda(); yt = torch.tensor(Y).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-4)
# 小数据:80/20
n = len(xt); perm = torch.randperm(n, generator=torch.Generator().manual_seed(7))
tr, va = perm[:int(n*0.8)], perm[int(n*0.8):]
best = 9
for ep in range(60):
    m.train()
    for i in range(0, len(tr), 16):
        b = tr[i:i+16]
        loss = nn.functional.smooth_l1_loss(m(xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        e = (m(xt[va]).cpu()-yt[va].cpu()).abs().mean(0)
    mae = e.tolist()
    print(f"ep{ep+1} h={mae[0]:.4f} w={mae[1]:.4f} g={mae[2]:.3f} mu={mae[3]:.3f} pr={mae[5]:.4f}", flush=True)
    if mae[0] < best:
        best = mae[0]
        torch.save(m.state_dict(), r"B:\mode\models\vyton_measure_v62.pt")
print("MV62_DONE h_mae:", round(best,4), flush=True)
