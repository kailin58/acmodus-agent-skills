
# Style v18盲测证据图(四头预测可视化)
import torch, torch.nn as nn, glob, os, json
import numpy as np
from PIL import Image, ImageDraw
from torchvision import models as tvm, transforms

class H(nn.Module):
    def __init__(s, fin=38400):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(fin, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 384), nn.ReLU())
        s.hs = nn.Linear(384, 8); s.he = nn.Linear(384, 6)
    def forward(s, x):
        h = s.trunk(x); return s.hs(h), s.he(h)
net = H(38400).cuda().eval()
net.load_state_dict(torch.load(r"B:\mode\style_train\vyton_style_v18_heads.pt", map_location="cuda"))
m = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.IMAGENET1K_V1)
m.classifier = nn.Identity()
bb = m.cuda().eval()
tf = transforms.Compose([transforms.Resize((160,160)), transforms.ToTensor()])
STYLES = ['casual','formal','street','vintage','minimalist','sporty','boho','elegant']
SCENES = ['work','casual','party','date','home','travel']
import random
random.seed(5)
pool = sorted(glob.glob(r"B:\mode\pinterest\outfit_imgs\*.jpg"))[-250:]
picks = random.sample(pool, 6)
rows = []
for p in picks:
    x = tf(Image.open(p).convert("RGB")).unsqueeze(0).cuda()
    with torch.no_grad():
        f = m.features(x).flatten(1)
        ps, pe = net(f)
        sprob = torch.sigmoid(ps)[0].cpu().numpy()
        eprob = torch.sigmoid(pe)[0].cpu().numpy()
    im = Image.open(p).convert("RGB").resize((240, 240))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 240, 34], fill=(0,0,0))
    txt = " ".join([f"{STYLES[i]}:{sprob[i]:.2f}" for i in np.argsort(-sprob)[:2]])
    d.text((4, 4), txt[:34], fill=(255,255,0))
    d.rectangle([0, 206, 240, 240], fill=(0,0,0))
    txt2 = " ".join([f"{SCENES[i]}:{eprob[i]:.2f}" for i in np.argsort(-eprob)[:2]])
    d.text((4, 210), txt2[:34], fill=(0,255,255))
    rows.append(np.asarray(im))
grid = np.concatenate([np.concatenate([rows[i], rows[i+1]], axis=1) for i in range(0, 6, 2)], axis=0)
Image.fromarray(grid).save(r"B:\mode\style_train\v18_blind_evidence.png")
print("V18_BLIND_OK", grid.shape)
