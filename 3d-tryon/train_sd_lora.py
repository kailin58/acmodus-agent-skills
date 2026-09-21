
# gen-base换装light-adapter训练 v1 (3060Ti 8G)
import os, io, json, math, time
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import AdapterConfig, get_adapter_model

SD = r"B:\mode\models\sd15\v1-5-pruned-emaonly.safetensors"
DATA = r"B:\mode\vton_training\sd_data"
OUT = r"B:\mode\vton_training\sd_ladapter_v1"
DEVICE = "cuda"
BS, GS = 1, 2   # batch1 梯度累积2 (8G显存)
STEPS = 300

# 数据
caps = [l.split("\t") for l in open(os.path.join(DATA,"captions.tsv"),encoding="utf-8").read().splitlines()]
pairs = [(os.path.join(DATA,p), c) for p,c in caps]

class DS(Dataset):
    def __init__(self, pipe):
        self.pipe = pipe; self.items = pairs
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, c = self.items[i % len(self.items)]
        img = Image.open(p).convert("RGB")
        pt = self.pipe.feature_extractor(img, return_tensors="pt").pixel_values[0]
        ci = self.pipe.tokenizer(c, max_length=77, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
        return pt.squeeze(), ci

def main():
    pipe = StableDiffusionPipeline.from_single_file(SD, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    pipe.to(DEVICE)
    vae, unet, te = pipe.vae, pipe.unet, pipe.tokenizer, # keep
    # light-adapter on UNet
    lcfg = AdapterConfig(r=8, ladapter_alpha=16, target_modules=["to_q","to_k","to_v","to_out_0"], ladapter_dropout=0.0)
    global unet_l
    unet_l = get_adapter_model(pipe.unet, lcfg)
    unet_l.print_trainable_parameters()
    for p in pipe.vae.parameters(): p.requires_grad_(False)
    for p in pipe.text_encoder.parameters(): p.requires_grad_(False)

    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)
    ds = DS(pipe)
    dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
    opt = torch.optim.AdamW([p for p in unet_l.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    unet_l.train(); vae.eval(); pipe.text_encoder.eval()
    step, t0 = 0, time.time()
    log = open(os.path.join(DATA, "train_log.txt"), "a", encoding="utf-8")
    while step < STEPS:
        for pt, ci in dl:
            pt = pt.to(DEVICE, dtype=torch.float16); ci = ci.to(DEVICE)
            with torch.no_grad():
                lat = vae.encode(pt).latent_dist.sample() * 0.18215
                enc = pipe.text_encoder(ci)[0]
            nz = torch.randn_like(lat)
            ts = torch.randint(0, 1000, (lat.shape[0],), device=DEVICE).long()
            snz = noise_sched.add_noise(lat, nz, ts)
            with torch.autocast("cuda", torch.float16):
                pred = unet_l(snz, ts, enc).sample
                loss = torch.nn.functional.mse_loss(pred.float(), nz.float())
            scaler.scale(loss / GS).backward()
            if (step+1) % GS == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            step += 1
            if step % 20 == 0:
                msg = f"step {step}/{STEPS} loss {loss.item():.4f} {step/(time.time()-t0):.2f}it/s"
                print(msg); log.write(msg+"\n"); log.flush()
            if step >= STEPS: break
    unet_l.save_pretrained(OUT)
    msg = f"LORA_SAVED {OUT}"
    print(msg); log.write(msg+"\n"); log.close()

if __name__ == "__main__":
    main()
