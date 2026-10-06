# 教师SDXL v3:脸部crop放大2x+高strength毛孔(终极pass)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, glob
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe = pipe.to("cuda")

srcs = sorted(glob.glob("/root/autodl-tmp/teacher_sdxl_v2/tv2_*.png"))
print("src:", len(srcs), flush=True)
os.makedirs("/root/autodl-tmp/teacher_sdxl_v3", exist_ok=True)
for i, p in enumerate(srcs):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    face = im.crop((0, 0, w, int(h*0.5)))
    face = face.resize((1024, 1024))  # 放大2x=毛孔像素空间
    out = pipe(prompt="(RAW photo of face:1.2), visible skin pores, skin micro-texture, fine vellus hair, subsurface scattering, film grain, 8k uhd",
               negative_prompt="airbrushed, smooth skin, plastic skin, retouched, cartoon, anime, blurry, waxy",
               image=face, strength=0.65, guidance_scale=7.0, num_inference_steps=28).images[0]
    out = out.resize((w, int(h*0.5)), Image.LANCZOS)
    merged = im.copy()
    merged.paste(out, (0, 0))
    merged.save(f"/root/autodl-tmp/teacher_sdxl_v3/tv3_{i:04d}.png")
    if i % 20 == 0: print(i, flush=True)
print("TEACHER_V3_DONE", flush=True)
