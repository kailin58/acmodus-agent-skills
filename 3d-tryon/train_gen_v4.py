
# SD教师v4:修裁头(负向prompt固化进采样脚本+数据清洗版重训)
# v4策略:用v3数据(3000对,head visible标注)但训练时降低步数防过拟合电商裁头先验+采样负向词
import os, io, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import AdapterConfig, get_adapter_model

SD = r"B:\mode\models\sd15\v1-5-pruned-emaonly.safetensors"
DATA = r"B:\mode\vton_training\sd_data_v3"
OUT = r"B:\mode\vton_training\sd_lora_v4"

caps = [l.split("\t") for l in open(os.path.join(DATA,"captions.tsv"),encoding="utf-8").read().splitlines()]
pairs = [(os.path.join(DATA,p), c) for p,c in caps][:2000]  # 2000对防过拟合
print("数据:", len(pairs), flush=True)

pipe = StableDiffusionPipeline.from_single_file(SD, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe.to("cuda")
lcfg = AdapterConfig(r=8, adapter_alpha=12, target_modules=["to_q","to_k","to_v","to_out_0"], adapter_dropout=0.05)  # alpha降+dropout加:防过拟合
unet_l = get_peft_model(pipe.unet, lcfg)
for p in pipe.vae.parameters(): p.requires_grad_(False)
for p in pipe.text_encoder.parameters(): p.requires_grad_(False)

sched = DDPMScheduler.from_config(pipe.scheduler.config)
class DS(Dataset):
    def __init__(self): self.items = pairs
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, c = self.items[i % len(self.items)]
        img = Image.open(p).convert("RGB")
        pt = pipe.feature_extractor(img, return_tensors="pt").pixel_values[0]
        ci = pipe.tokenizer(c, max_length=77, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
        return pt.squeeze(), ci

dl = DataLoader(DS(), batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
opt = torch.optim.AdamW([p for p in unet_l.parameters() if p.requires_grad], lr=8e-5)  # lr降
scaler = torch.amp.GradScaler("cuda")
unet_l.train(); pipe.vae.eval(); pipe.text_encoder.eval()
step, t0 = 0, None
log = open(os.path.join(r"B:\mode\vton_training", "v4_log.txt"), "a", encoding="utf-8")
while step < 800:
    for pt, ci in dl:
        pt = pt.to("cuda", dtype=torch.float16); ci = ci.to("cuda")
        with torch.no_grad():
            lat = pipe.vae.encode(pt).latent_dist.sample()*0.18215
            enc = pipe.text_encoder(ci)[0]
        nz = torch.randn_like(lat)
        ts = torch.randint(0,1000,(lat.shape[0],),device="cuda").long()
        snz = sched.add_noise(lat,nz,ts)
        with torch.autocast("cuda", torch.float16):
            pred = unet_l(snz,ts,enc).sample
            loss = torch.nn.functional.mse_loss(pred.float(), nz.float())
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad()
        step += 1
        if step % 50 == 0:
            msg = f"v4 step {step}/800 loss {loss.item():.4f}"
            print(msg, flush=True); log.write(msg+"\n"); log.flush()
        if step >= 800: break
unet_l.save_pretrained(OUT)
print("V4_SAVED", flush=True)
