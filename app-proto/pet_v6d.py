import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, math, glob
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image, ImageDraw, ImageFilter, ImageOps

cn = ControlNetModel.from_single_file(r"B:\mode\models\control_v11p_sd15_canny.pth", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_single_file(
    r"B:\mode\models\sd15_pruned_emaonly.safetensors",
    controlnet=cn, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
print("PIPE READY", flush=True)

def knit_canny(size=512):
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    for x in range(0, size, 20):
        d.line([(x, 0), (x, size)], fill=100, width=4)
    for cx in [size*0.26, size*0.5, size*0.74]:
        y = 36
        while y < size - 36:
            amp = 32 * math.sin((y/size) * math.pi)
            d.arc([cx-amp-14, y, cx+amp-14+28, y+44], start=200 if (y//44)%2==0 else 20, end=380, fill=255, width=9)
            y += 44
    return im.filter(ImageFilter.GaussianBlur(0.6))

srcs = sorted(glob.glob(r"B:\mode\pet\lora_data_v3\dog_*_src.png"))[:6]
os.makedirs(r"B:\mode\pet_out6d", exist_ok=True)
for i, p in enumerate(srcs):
    im0 = Image.open(p).convert("L").resize((456, 440))
    canvas = Image.new("L", (512, 512), 255)
    canvas.paste(im0, (28, 46))
    base = ImageOps.autocontrast(canvas, cutoff=4)
    edges = base.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(5)).point(lambda v: min(int(v*2.2), 230))
    Hmask = Image.new("L", (512, 512), 50)
    d2 = ImageDraw.Draw(Hmask)
    d2.ellipse([150, 30, 380, 250], fill=210)
    edges = Image.composite(edges, edges.point(lambda v: int(v*0.35)), Hmask)
    guide = Image.blend(edges, knit_canny(), 0.55)
    out = pipe("(a cute dog wearing a thick chunky cable-knit sweater:1.2), dog centered full body, plain gray studio background, (prominent braided cable knit pattern:1.3)",
               image=guide, num_inference_steps=34, guidance_scale=8.5, controlnet_conditioning_scale=1.1,
               negative_prompt="deformed, extra limbs, cropped, embedded in fabric, blurry, watermark, smooth fabric").images[0]
    out.save(rf"B:\mode\pet_out6d\pet_d6_{i:02d}.png")
    print("done", i, flush=True)
print("PET_V6D_DONE", flush=True)