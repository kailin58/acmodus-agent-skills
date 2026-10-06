# v12s 100% crop终判图
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms.functional as TF

class RegUNet(nn.Module):
    def __init__(s, ch=48):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.GroupNorm(8,o), nn.SiLU(), nn.Conv2d(o,o,3,padding=1), nn.GroupNorm(8,o), nn.SiLU())
        s.d1 = blk(6, ch); s.d2 = blk(ch, ch*2); s.d3 = blk(ch*2, ch*3)
        s.mid = blk(ch*3, ch*3)
        s.u3 = blk(ch*6, ch*2); s.u2 = blk(ch*4, ch); s.u1 = blk(ch*2, ch)
        s.out = nn.Conv2d(ch, 3, 1)
        s.hd_gender = nn.Linear(ch, 2)
        s.hd_measure = nn.Linear(ch, 6)
        s.pool = nn.MaxPool2d(2)
    def forward(s, inp):
        h1 = s.d1(inp); h2 = s.d2(s.pool(h1)); h3 = s.d3(s.pool(h2))
        m = s.mid(s.pool(h3))
        u3 = s.u3(torch.cat([nn.functional.interpolate(m, scale_factor=2, mode="nearest"), h3],1))
        u2 = s.u2(torch.cat([nn.functional.interpolate(u3, scale_factor=2, mode="nearest"), h2],1))
        u1 = s.u1(torch.cat([nn.functional.interpolate(u2, scale_factor=2, mode="nearest"), h1],1))
        f = u1.mean((2,3))
        return torch.tanh(s.out(u1)), s.hd_gender(f), s.hd_measure(f)

net = RegUNet().cuda()
net.load_state_dict(torch.load(r"B:\mode\models\vyton_try_student_v12s_tex12.pt", map_location="cuda", weights_only=True))
net.eval()
src = Image.open(r"B:\mode\vton_training\teacher_pairs_v2\t_0250.png").convert("RGB")
W, H = src.size
img = src.crop((0, 0, W, int(H*0.62))).resize((192,192))
x = torch.tensor(np.asarray(img)/127.5-1, dtype=torch.float32).permute(2,0,1)
blur = TF.gaussian_blur(x, [9,9])
cond = TF.resize(x[:, 60:140, 70:130], (192,192))
inp = torch.cat([blur, cond], 0).unsqueeze(0).cuda()
with torch.no_grad():
    out = net(inp)[0][0].cpu()
# 三区100%crop放大4x:脸颊/发际线/条纹边(上2x2拼)
def crop4x(t, y0, x0, h=48, w=48):
    c = t[:, y0:y0+h, x0:x0+w]
    return TF.resize(c, (h*4, w*4), interpolation=TF.InterpolationMode.NEAREST)
cheek = crop4x(out, 70, 80)     # 脸颊毛孔区
hair = crop4x(out, 30, 90)      # 发际线
stripe = crop4x(out, 130, 60)   # 条纹边界
top = torch.cat([cheek, hair], 2)
bot = torch.cat([stripe, crop4x(out, 150, 60)], 2)
grid = torch.cat([top, bot], 1)
gimg = ((grid.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype("uint8")
Image.fromarray(gimg).save(r"B:\mode\vton_training\student_v12s_crops.png")
print("CROPS_DONE")
