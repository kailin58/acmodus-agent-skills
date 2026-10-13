# 层2·分辨率:v7 512→768→1024两段低denoise放大
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

BASE = r"B:\mode\vton_training"
pipe = StableDiffusionImg2ImgPipeline.from_single_file(
    r"B:\mode\models\sd15\v1-5-pruned-emaonly.safetensors",
    torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.enable_attention_slicing()

cur = Image.open(BASE + r"\e2e_hand_v7.png").convert("RGB")
PROMPT = "professional photo, hand with french manicure, rose print clothing, natural skin texture, photorealistic, sharp focus, high detail"
NEG = "blurry, melting, plastic, wax, deformed, low quality"

# 段1: 512→768
g = torch.Generator("cuda").manual_seed(201)
mid = pipe(prompt=PROMPT, negative_prompt=NEG,
           image=cur.resize((768, 768), Image.LANCZOS),
           strength=0.25, guidance_scale=6.0,
           num_inference_steps=22, generator=g).images[0]
mid.save(BASE + r"\e2e_hand_v8_768.png")
print("段1 768完成", flush=True)

# 段2: 768→1024
g2 = torch.Generator("cuda").manual_seed(202)
hi = pipe(prompt=PROMPT, negative_prompt=NEG,
          image=mid.resize((1024, 1024), Image.LANCZOS),
          strength=0.22, guidance_scale=6.0,
          num_inference_steps=24, generator=g2).images[0]
hi.save(BASE + r"\e2e_hand_v8.png")
print("V8_RES_DONE", flush=True)
