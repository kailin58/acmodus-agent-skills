# 学生v6采样验证
import torch, torch.nn as nn
import numpy as np
from PIL import Image

class UNet(nn.Module):
    def __init__(s, ch=48):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.GroupNorm(8,o), nn.SiLU(), nn.Conv2d(o,o,3,padding=1), nn.GroupNorm(8,o), nn.SiLU())
        s.d1 = blk(3+1, ch); s.d2 = blk(ch, ch*2); s.d3 = blk(ch*2, ch*3)
        s.mid = blk(ch*3, ch*3)
        s.u3 = blk(ch*3+ch*3, ch*2); s.u2 = blk(ch*2+ch*2, ch); s.u1 = blk(ch+ch, ch)
        s.out = nn.Conv2d(ch, 3, 1)
        s.pool = nn.MaxPool2d(2)
    def forward(s, x, t_emb):
        t = t_emb.view(-1,1,1,1).expand(-1,1,x.shape[2],x.shape[3])
        h1 = s.d1(torch.cat([x,t],1)); h2 = s.d2(s.pool(h1)); h3 = s.d3(s.pool(h2))
        m = s.mid(s.pool(h3))
        u3 = s.u3(torch.cat([nn.functional.interpolate(m, scale_factor=2, mode="nearest"), h3],1))
        u2 = s.u2(torch.cat([nn.functional.interpolate(u3, scale_factor=2, mode="nearest"), h2],1))
        u1 = s.u1(torch.cat([nn.functional.interpolate(u2, scale_factor=2, mode="nearest"), h1],1))
        return s.out(u1)

net = UNet().cuda()
net.load_state_dict(torch.load(r"B:\mode\models\vyton_try_student_v6g.pt", map_location="cuda", weights_only=True))
net.eval()
T = 400
betas = torch.linspace(1e-4, 0.02, T).cuda()
abar = torch.cumprod(1-betas, 0).cuda()
g = torch.Generator("cuda").manual_seed(9)
x = torch.randn(1,3,128,128, device="cuda", generator=g)
with torch.no_grad():
    for t in reversed(range(T)):
        tt = torch.tensor([t/T], device="cuda")
        pred = net(x, tt)
        alpha = 1-betas[t]; ab = abar[t]
        x = (1/alpha.sqrt())*(x - (1-alpha)/(1-ab).sqrt()*pred)
        if t > 0:
            x = x + betas[t].sqrt()*torch.randn(1,3,128,128, device="cuda", generator=torch.Generator("cuda").manual_seed(t))
img = ((x[0].cpu().numpy().transpose(1,2,0)+1)*127.5).clip(0,255).astype("uint8")
Image.fromarray(img).save(r"B:\mode\vton_training\student_v6g_sample.png")
print("SAMPLE_DONE")
