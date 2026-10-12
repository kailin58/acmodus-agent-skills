
# cut_v2 vs u2net同域对比+软边指标
import torch, glob, os
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, r"B:\mode\style_train")

# 重建结构(同train_v2)
import torch.nn as nn
from torchvision import models as tvm, transforms
class BackboneV2(nn.Module):
    def __init__(s):
        super().__init__()
        m = tvm.mobilenet_v3_small(weights=None)
        s.feat = m.features
    def forward(s, x): return s.feat(x)
class HeadV2(nn.Module):
    def __init__(s, inch=576):
        super().__init__()
        s.up = nn.Sequential(
            nn.Conv2d(inch, 64, 1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, 3, padding=1))
    def forward(s, f): return s.up(f)

bb, hd = BackboneV2(), HeadV2()
bb.load_state_dict(torch.load(r"B:\mode\models\cut\vyton_cut_v2_backbone.pt", map_location="cpu"))
hd.load_state_dict(torch.load(r"B:\mode\models\cut\vyton_cut_v2_head.pt", map_location="cpu"))
net = torch.nn.Sequential(bb, hd).eval()
tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# u2net int8
import onnxruntime as ort
sess = ort.InferenceSession(r"B:\mode\models\cut\vyton_cut_int8.onnx", providers=["CPUExecutionProvider"])
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)

# 测试集=验证split同款(尾5%)
pseudo = sorted(glob.glob(r"B:\mode\models\cut\pseudo_labels\*.png"))
pairs = []
for m in pseudo:
    name = os.path.basename(m)[:-4]
    for base in [r"B:\mode\pinterest\outfit_imgs", r"B:\mode\yt_outfit", r"B:\mode\bz_outfit",
                 r"B:\mode\ig_outfit\cells", r"B:\mode\pinterest\board_imgs",
                 r"B:\mode\glow_data\mt\extracted\makeup\MAKE",
                 r"B:\mode\glow_data\mt\extracted\non-makeup\non", r"B:\mode\pet\lora_data_v3"]:
        src = os.path.join(base, name)
        if os.path.exists(src):
            pairs.append((src, m)); break
test = pairs[int(len(pairs)*0.95):][:60]
print(f"同域测试{len(test)}张", flush=True)

iou_v2, iou_u2 = 0, 0
soft_v2, soft_u2 = 0, 0
n = 0
for src, mskf in test:
    img = Image.open(src).convert("RGB")
    W, H = img.size
    gt = np.asarray(Image.open(mskf).convert("L").resize((W, H)), np.float32)/255
    # v2
    with torch.no_grad():
        x = tf(img).unsqueeze(0)
        p2 = torch.sigmoid(net(x))[0,0].numpy()
    p2 = np.asarray(Image.fromarray((p2*255).astype(np.uint8)).resize((W, H)), np.float32)/255
    # u2net
    im = img.resize((320,320))
    xa = (np.asarray(im, np.float32)/255 - MEAN)/STD
    xa = xa.transpose(2,0,1)[None]
    u2 = sess.run(None, {sess.get_inputs()[0].name: xa})[0][0][0]
    u2 = np.asarray(Image.fromarray(((u2>0.5).astype(np.uint8)*255)).resize((W,H)), np.float32)/255
    # IoU(对BiRefNet教师GT)
    b2 = p2 > 0.5
    i2 = ((b2 & (gt>0.5)).sum()) / max(((b2 | (gt>0.5)).sum()), 1)
    bu = u2 > 0.5
    iu = ((bu & (gt>0.5)).sum()) / max(((bu | (gt>0.5)).sum()), 1)
    iou_v2 += i2; iou_u2 += iu
    soft_v2 += ((p2>0.1)&(p2<0.9)).mean()
    soft_u2 += ((u2>0.1)&(u2<0.9)).mean()
    n += 1
print(f"cut_v2: IoU{iou_v2/n:.4f} 软边{soft_v2/n*100:.2f}%", flush=True)
print(f"u2net : IoU{iou_u2/n:.4f} 软边{soft_u2/n*100:.2f}%", flush=True)
