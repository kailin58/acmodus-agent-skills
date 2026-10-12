
# BiRefNet官方加载法(load_from_pretrained自动下swin主干)
import os, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.chdir(r"B:\mode\third_party\BiRefNet")
sys.path.insert(0, r"B:\mode\third_party\BiRefNet")
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.birefnet import BiRefNet

t = BiRefNet.load_pretrained(r"B:\mode\models\cut\teacher_birefnet\model.safetensors") if hasattr(BiRefNet, "load_pretrained") else None
if t is None:
    t = BiRefNet.from_pretrained("ZhengPeng7/BiRefNet")
t = t.cuda().eval()
print("LOADED", flush=True)
import glob
pool = (glob.glob(r"B:\mode\pinterest\outfit_imgs\*.jpg")[:20] +
        glob.glob(r"B:\mode\yt_outfit\*.jpg")[:15] +
        glob.glob(r"B:\mode\glow_data\mt\extracted\makeup\MAKE\*.jpg")[:10] +
        glob.glob(r"B:\mode\pet\lora_data_v3\dog_*_src.png")[:5])
os.makedirs(r"B:\mode\models\cut\pseudo_labels", exist_ok=True)
tr = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
n = 0
with torch.no_grad():
    for f in pool:
        img = Image.open(f).convert("RGB")
        x = tr(img).unsqueeze(0).cuda()
        pred = t(x)[-1].sigmoid().cpu()[0, 0]
        Image.fromarray((pred*255).numpy().astype(np.uint8)).resize(img.size, Image.BILINEAR).save(rf"B:\mode\models\cut\pseudo_labels\{os.path.basename(f)}.png")
        n += 1
print("PSEUDO_50_DONE", n, flush=True)
