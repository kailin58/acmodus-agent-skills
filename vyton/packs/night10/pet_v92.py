# v9.2:下摆罗纹规则化+侧面浮雕(窄带精修)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionInpaintPipeline

BASE = r"B:/mode/pet_out8"
SD = r"B:/mode/models/sd15/v1-5-pruned-emaonly.safetensors"
img = Image.open(BASE + "/v9_1.png").convert("RGB")
W, H = img.size
# 下摆带+左右侧条
mask = Image.new("L", (W, H), 0)
dm = ImageDraw.Draw(mask)
dm.rounded_rectangle([int(W*0.20), int(H*0.62), int(W*0.84), int(H*0.75)], radius=20, fill=255)  # 下摆
dm.rounded_rectangle([int(W*0.16), int(H*0.35), int(W*0.26), int(H*0.68)], radius=20, fill=255)   # 左侧
dm.rounded_rectangle([int(W*0.76), int(H*0.35), int(W*0.88), int(H*0.68)], radius=20, fill=255)   # 右侧
mask = mask.filter(ImageFilter.GaussianBlur(8))
pipe = StableDiffusionInpaintPipeline.from_single_file(
    SD, torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.enable_attention_slicing()
g = torch.Generator("cuda").manual_seed(555)
out = pipe(
    prompt="chunky cable knit sweater, neat vertical ribbed hem with blue stripe trim, consistent knitting pattern on sides, photorealistic",
    negative_prompt="melting hem, irregular ribbing, flat texture, blur, deformed body",
    image=img.resize((512, 512)),
    mask_image=mask.resize((512, 512)),
    num_inference_steps=30,
    guidance_scale=7.0,
    strength=0.55,
    generator=g,
).images[0]
out.save(BASE + "/v9_2.png")
print("V92_DONE")
