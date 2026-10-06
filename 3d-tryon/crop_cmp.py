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
        s.hd_gender = nn.Linear(ch, 2); s.hd_measure = nn.Linear(ch, 6)
        s.pool = nn.MaxPool2d(2)
    def forward(s, inp):
        h1 = s.d1(inp); h2 = s.d2(s.pool(h1)); h3 = s.d3(s.pool(h2))
        m = s.mid(s.pool(h3))
        u3 = s.u3(torch.cat([nn.functional.interpolate(m, scale_factor=2, mode="nearest"), h3],1))
        u2 = s.u2(torch.cat([nn.functional.interpolate(u3, scale_factor=2, mode="nearest"), h2],1))
        u1 = s.u1(torch.cat([nn.functional.interpolate(u2, scale_factor=2, mode="nearest"), h1],1))
        f = u1.mean((2,3))
        return torch.tanh(s.out(u1)), s.hd_gender(f), s.hd_measure(f)

src = Image.open(r"B:\mode\vton_training\teacher_pairs_v2\t_0250.png").convert("RGB")
W, H = src.size
half = src.crop((0, 0, W, int(H*0.62)))
outs = {}
for name, size, ckpt in [("192", 192, r"B:\mode\models\vyton_try_student_v11r_edge.pt"), ("256", 256, r"B:\mode\models\vyton_try_student_v16_256.pt")]:
    img = half.resize((size, size))
    x = torch.tensor(np.asarray(img)/127.5-1, dtype=torch.float32).permute(2,0,1)
    blur = TF.gaussian_blur(x, [9,9])
    cond = TF.resize(x[:, int(size*0.31):int(size*0.73), int(size*0.36):int(size*0.64)], (size, size))
    inp = torch.cat([blur, cond], 0).unsqueeze(0).cuda()
    net = RegUNet().cuda()
    sd = torch.load(ckpt, map_location="cuda", weights_only=True)
    net.load_state_dict({k: v for k, v in sd.items() if k in net.state_dict()}, strict=False)
    net.eval()
    with torch.no_grad():
        out = net(inp)[0][0].cpu()
    y0, y1 = int(size*0.36), int(size*0.60)
    x0, x1 = int(size*0.40), int(size*0.63)
    crop = out[:, y0:y1, x0:x1]
    outs[name] = TF.resize(crop, (256, 256), interpolation=TF.InterpolationMode.BICUBIC)
    del net
    torch.cuda.empty_cache()
grid = torch.cat([outs["192"], outs["256"]], 2)
g = ((grid.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype("uint8")
Image.fromarray(g).save(r"B:\mode\vton_training\v16_vs_v11r_crop.png")
print("CROP_CMP_DONE")