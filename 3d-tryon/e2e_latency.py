import time, torch
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, r"B:\mode\pinterest\acmodus-agent-skills\3d-tryon")
from vyton_pore_head import pore_head

class RegUNet(torch.nn.Module):
    def __init__(s, ch=48):
        super().__init__()
        def blk(i,o): return torch.nn.Sequential(torch.nn.Conv2d(i,o,3,padding=1), torch.nn.GroupNorm(8,o), torch.nn.SiLU(), torch.nn.Conv2d(o,o,3,padding=1), torch.nn.GroupNorm(8,o), torch.nn.SiLU())
        s.d1 = blk(6, ch); s.d2 = blk(ch, ch*2); s.d3 = blk(ch*2, ch*3)
        s.mid = blk(ch*3, ch*3)
        s.u3 = blk(ch*6, ch*2); s.u2 = blk(ch*4, ch); s.u1 = blk(ch*2, ch)
        s.out = torch.nn.Conv2d(ch, 3, 1)
        s.hd_gender = torch.nn.Linear(ch, 2); s.hd_measure = torch.nn.Linear(ch, 6)
        s.pool = torch.nn.MaxPool2d(2)
    def forward(s, inp):
        h1 = s.d1(inp); h2 = s.d2(s.pool(h1)); h3 = s.d3(s.pool(h2))
        m = s.mid(s.pool(h3))
        u3 = s.u3(torch.cat([torch.nn.functional.interpolate(m, scale_factor=2, mode="nearest"), h3],1))
        u2 = s.u2(torch.cat([torch.nn.functional.interpolate(u3, scale_factor=2, mode="nearest"), h2],1))
        u1 = s.u1(torch.cat([torch.nn.functional.interpolate(u2, scale_factor=2, mode="nearest"), h1],1))
        f = u1.mean((2,3))
        return torch.tanh(s.out(u1)), s.hd_gender(f), s.hd_measure(f)

net = RegUNet().cuda()
sd = torch.load(r"B:\mode\models\vyton_try_student_v17.pt", map_location="cuda", weights_only=True)
net.load_state_dict({k: v for k, v in sd.items() if k in net.state_dict()}, strict=False)
net.eval()
x = torch.randn(1, 6, 256, 256).cuda()
# GPU热身
for _ in range(3): net(x)
torch.cuda.synchronize()
t0 = time.time()
N = 20
for _ in range(N): net(x)
torch.cuda.synchronize()
gpu_ms = (time.time()-t0)/N*1000
# CPU延迟(端侧代理)
netc = net.cpu(); xc = x.cpu()
t0 = time.time()
for _ in range(3): netc(xc)
cpu_ms = (time.time()-t0)/3*1000
# 毛孔头延迟
img = Image.new("RGB", (256,256), (200,160,130))
t0 = time.time()
for _ in range(5): pore_head(img)
pore_ms = (time.time()-t0)/5*1000
print(f"学生GPU={gpu_ms:.1f}ms CPU={cpu_ms:.0f}ms 毛孔头={pore_ms:.0f}ms 合计端侧≈{cpu_ms+pore_ms:.0f}ms")
import torch as T
n_par = sum(p.numel() for p in net.parameters())/1e6
print(f"参数{n_par:.1f}M 显存占用(fp32)={n_par*4:.0f}MB")