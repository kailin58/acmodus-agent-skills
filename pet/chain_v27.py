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
d.rounded_rectangle((100, 100, 430, 512), radius=50, fill=255)  # v25:上缘提到y100(盖背肩)
# 头保护(上部圆)
hd = Image.new("L",(512,512),255)
ImageDraw.Draw(hd).ellipse((140,-160,400,160), fill=0)  # v25:头保护只到y160,收紧x
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
pipe.unet = PeftModel.from_pretrained(pipe.unet, r"B:\mode\pet\sd_lora_pet_v22")
pipe.set_progress_bar_config(disable=True)
out = pipe("a dog wearing a full-coverage thick red (chunky cable-knit sweater:1.4) with (twisted rope cables:1.3) and ribbed knit texture, covering chest, belly and back extending down to its belly, ribbed hem",
           image=img, mask_image=mask,
           strength=0.82, num_inference_steps=32, guidance_scale=7.0,
           negative_prompt="bare fur, naked dog, drool, foam around mouth").images[0]
out.save(r"B:\mode\pet\v27_final.png")
print("V27_DONE", flush=True)
