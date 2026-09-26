# 宠物穿衣LoRA训练(50合成对x800步)
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model

SD = r"B:\mode\models\sd15\v1-5-pruned-emaonly.safetensors"
DATA = r"B:\mode\pet\lora_data_v2"
OUT = r"B:\mode\pet\sd_lora_pet_v2"
DEVICE = "cuda"
STEPS = 1200

caps = [l.split("\t") for l in open(os.path.join(DATA,"captions.tsv"),encoding="utf-8").read().splitlines()]
pairs = [(os.path.join(DATA,p), c.strip()) for p,c in caps]

class DS(Dataset):
    def __init__(self, pipe):
        self.pipe = pipe; self.items = pairs
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, c = self.items[i % len(self.items)]
        img = Image.open(p).convert("RGB").resize((512,512))
        pt = self.pipe.feature_extractor(img, return_tensors="pt").pixel_values[0]
        ci = self.pipe.tokenizer(c, max_length=77, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
        return pt.squeeze(), ci

pipe = StableDiffusionPipeline.from_single_file(SD, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe.to(DEVICE)
lcfg = LoraConfig(r=8, lora_alpha=16, target_modules=["to_q","to_k","to_v","to_out_0"], lora_dropout=0.0)
unet_l = get_peft_model(pipe.unet, lcfg)
unet_l.print_trainable_parameters()
for prm in pipe.vae.parameters(): prm.requires_grad_(False)
for prm in pipe.text_encoder.parameters(): prm.requires_grad_(False)
sched = DDPMScheduler.from_config(pipe.scheduler.config)
ds = DS(pipe)
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
opt = torch.optim.AdamW([p for p in unet_l.parameters() if p.requires_grad], lr=1e-4)
scaler = torch.amp.GradScaler("cuda")
unet_l.train(); pipe.vae.eval(); pipe.text_encoder.eval()
import time
step, t0 = 0, time.time()
while step < STEPS:
    for pt, ci in dl:
        pt = pt.to(DEVICE, dtype=torch.float16); ci = ci.to(DEVICE)
        with torch.no_grad():
            lat = pipe.vae.encode(pt).latent_dist.sample()*0.18215
            enc = pipe.text_encoder(ci)[0]
        nz = torch.randn_like(lat)
        ts = torch.randint(0,1000,(lat.shape[0],),device=DEVICE).long()
        snz = sched.add_noise(lat, nz, ts)
        with torch.autocast("cuda", torch.float16):
            pred = unet_l(snz, ts, enc).sample
            loss = torch.nn.functional.mse_loss(pred.float(), nz.float())
        scaler.scale(loss/2).backward()
        if (step+1)%2==0:
            scaler.step(opt); scaler.update(); opt.zero_grad()
        step += 1
        if step%100==0:
            print(f"step {step}/{STEPS} loss {loss.item():.4f} {(step/(time.time()-t0)):.2f}it/s", flush=True)
        if step>=STEPS: break
unet_l.save_pretrained(OUT)
print("PET_LORA_SAVED", flush=True)
