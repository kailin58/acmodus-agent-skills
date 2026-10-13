
# v7:指缝条带重绘(只重绘粘连条带,保其余)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from diffusers import StableDiffusionInpaintPipeline

BASE = r"B:\mode\vton_training"
img = Image.open(BASE + "\e2e_hand_v6_1.png").convert("RGB")
W, H = img.size
# 指缝条带(检测到的x239,y190,w17,h41 膨胀)
x0, y0, x1, y1 = 239-20, 190-15, 239+17+20, 190+41+30
mask = Image.new("L", (W, H), 0)
dm = ImageDraw.Draw(mask)
dm.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(5))

pipe = StableDiffusionInpaintPipeline.from_single_file(
    r"B:\mode\models\sd15\v1-5-pruned-emaonly.safetensors",
    torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.enable_attention_slicing()
g = torch.Generator("cuda").manual_seed(77)
w2 = 512
h2 = int(H/W*w2)
out = pipe(
    prompt="macro photo of gap between two fingers, clear separation line, dark shadow line between fingers, natural skin, photorealistic detail",
    negative_prompt="fused fingers, webbed skin, blurry, melting",
    image=img.resize((w2, h2)),
    mask_image=mask.resize((w2, h2)),
    num_inference_steps=24,
    guidance_scale=7.0,
    strength=0.85,
    generator=g,
).images[0]
out = out.resize((W, H), Image.LANCZOS)
out.save(BASE + "\e2e_hand_v7.png")
print("V7_DONE")
