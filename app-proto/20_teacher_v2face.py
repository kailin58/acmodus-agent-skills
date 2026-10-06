# 教师SDXL v2:脸部区域二次pass(毛孔注入)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, glob
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe = pipe.to("cuda")

srcs = sorted(glob.glob("/root/autodl-tmp/teacher_sdxl/t_*.png"))
print("src:", len(srcs), flush=True)
os.makedirs("/root/autodl-tmp/teacher_sdxl_v2", exist_ok=True)
for i, p in enumerate(srcs[:100]):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    # 上半区(脸所在)crop→增强→贴回
    face = im.crop((0, 0, w, int(h*0.5)))
    face = face.resize((1024, 512))
    out = pipe(prompt="RAW photo, face closeup, detailed skin pores, visible skin texture, fine peach fuzz, film grain, subsurface scattering",
               negative_prompt="airbrushed, smooth skin, plastic, retouched, cartoon, blurry",
               image=face, strength=0.58, guidance_scale=6.5, num_inference_steps=26).images[0]
    out = out.resize((w, int(h*0.5)))
    merged = im.copy()
    merged.paste(out, (0, 0))
    merged.save(f"/root/autodl-tmp/teacher_sdxl_v2/tv2_{i:04d}.png")
    if i % 20 == 0: print(i, flush=True)
print("TEACHER_V2_DONE", flush=True)
