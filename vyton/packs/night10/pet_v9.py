# 宠物v9:下半身渐隐根修(img2img限定重绘下半)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline

BASE = r"B:/mode/pet_out8"
# v8最好:01_s123_c1.1(7/10真穿衣后半渐隐)
src = BASE + "/v8_01_s123_c1.1.png"
SD = r"B:/mode/models/sd15/v1-5-pruned-emaonly.safetensors"

img = Image.open(src).convert("RGB")
W, H = img.size

# inpaint:下半2/3重绘(渐隐区),上半保v8
mask = Image.new("L", (W, H), 0)
dm = ImageDraw.Draw(mask)
dm.rectangle([0, int(H*0.35), W, H], fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(12))

pipe = StableDiffusionInpaintPipeline.from_single_file(
    SD, torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.enable_attention_slicing()

results = []
for seed, name in [(123, "v9_s123"), (456, "v9_s456"), (789, "v9_s789")]:
    g = torch.Generator("cuda").manual_seed(seed)
    out = pipe(
        prompt="a dog wearing a knitted sweater, full body visible, complete body, all four legs clearly defined, solid ground shadow, professional pet photography, studio lighting, photorealistic",
        negative_prompt="fading, fade out, transparent, ghost, blurry bottom, missing legs, dissolving, melting, half body, cropped",
        image=img.resize((512, 512)),
        mask_image=mask.resize((512, 512)),
        num_inference_steps=28,
        guidance_scale=7.5,
        generator=g,
    ).images[0]
    out.save(BASE + f"/{name}.png")
    results.append(name)
    print(name, flush=True)
print("PET_V9_DONE")
