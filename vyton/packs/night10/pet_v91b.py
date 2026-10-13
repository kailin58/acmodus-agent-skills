# v9.1:789底+毛衣细节强化+头顶修复
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionInpaintPipeline

BASE = r"B:/mode/pet_out8"
SD = r"B:/mode/models/sd15/v1-5-pruned-emaonly.safetensors"
img = Image.open(BASE + "/v9_s789.png").convert("RGB")
W, H = img.size

# 毛衣区(躯干中部)材质增强重绘
mask = Image.new("L", (W, H), 0)
dm = ImageDraw.Draw(mask)
dm.rounded_rectangle([int(W*0.18), int(H*0.30), int(W*0.85), int(H*0.72)], radius=40, fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(10))

pipe = StableDiffusionInpaintPipeline.from_single_file(
    SD, torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.enable_attention_slicing()
g = torch.Generator("cuda").manual_seed(321)
out = pipe(
    prompt="dog wearing a chunky knitted sweater with detailed cable knit texture, ribbed collar and cuffs, visible wool fibers, photorealistic pet fashion photography",
    negative_prompt="smooth fur replacing sweater, plain fabric, low detail, blurry, deformed",
    image=img.resize((512, 512)),
    mask_image=mask.resize((512, 512)),
    num_inference_steps=30,
    guidance_scale=7.5,
    strength=0.65,
    generator=g,
).images[0]
out.save(BASE + "/v9_1.png")
print("V91_DONE")
