# v14:固定大矩形mask一步到位
import io
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import onnxruntime as ort
import torch
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel

# 1) 图(同链选图逻辑,取v11选中那张——直接读chain_src.png)
img = Image.open(r"B:\mode\pet\chain_src.png").convert("RGB")
w0, h0 = img.size
s = min(w0, h0)
img = img.crop(((w0-s)//2,(h0-s)//2,(w0+s)//2,(h0+s)//2)).resize((512,512))

# 2) 固定大mask:躯干+y方向大覆盖(头在上部保护)
mask = Image.new("L",(512,512),0)
d = ImageDraw.Draw(mask)
d.rounded_rectangle((110, 265, 420, 512), radius=45, fill=255)  # 躯干大区
# 头保护(上部圆)
hd = Image.new("L",(512,512),255)
ImageDraw.Draw(hd).ellipse((120,-110,420,230), fill=0)
hd = hd.filter(ImageFilter.GaussianBlur(15))
m = np.minimum(np.asarray(mask,np.float32), np.asarray(hd,np.float32))
mask = Image.fromarray(m.astype(np.uint8))
mask.save(r"B:\mode\pet\v14_mask.png")
print("mask白区:", round((np.asarray(mask)>128).mean()*100,1), "%", flush=True)

# 3) LoRA inpaint
pipe = StableDiffusionInpaintPipeline.from_single_file(
    r"B:\mode\models\sd15\v1-5-pruned-emaonly.safetensors",
    torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe.to("cuda")
from diffusers import StableDiffusionPipeline
pipe.unet = PeftModel.from_pretrained(pipe.unet, r"B:\mode\pet\sd_lora_pet_v2")
pipe.set_progress_bar_config(disable=True)
out = pipe("a dog wearing a full-coverage thick red knitted sweater extending down to its belly, ribbed hem",
           image=img, mask_image=mask,
           strength=0.82, num_inference_steps=32, guidance_scale=7.0,
           negative_prompt="bare fur, naked dog").images[0]
out.save(r"B:\mode\pet\v18_final.png")
print("V14_DONE", flush=True)
