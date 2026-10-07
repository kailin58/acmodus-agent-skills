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
    for x in range(0, size, 22):
        d.line([(x, 0), (x, size)], fill=90, width=4)
    for cx in [size*0.28, size*0.5, size*0.72]:
        y = 40
        while y < size - 40:
            amp = 30 * math.sin((y/size) * math.pi)
            d.arc([cx-amp-12, y, cx+amp-12+24, y+40], start=200 if (y//40)%2==0 else 20, end=380, fill=255, width=8)
            y += 40
    return im.filter(ImageFilter.GaussianBlur(0.7))

srcs = sorted(glob.glob(r"B:\mode\pet\lora_data_v3\dog_*_src.png"))[:6]
os.makedirs(r"B:\mode\pet_out6c", exist_ok=True)
for i, p in enumerate(srcs):
    im0 = Image.open(p).convert("L").resize((456, 440))
    canvas = Image.new("L", (512, 512), 255)
    canvas.paste(im0, (28, 46))
    base = ImageOps.autocontrast(canvas, cutoff=8)
    # 头肩区轮廓强,躯干让位给麻花
    edges = base.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3)).point(lambda v: min(int(v*1.8), 200))
    Hmask = Image.new("L", (512, 512), 60)  # 全图弱轮廓
    d = ImageDraw.Draw(Hmask)
    d.ellipse([150, 30, 380, 250], fill=200)  # 头区强
    edges = Image.composite(edges, edges.point(lambda v: int(v*0.4)), Hmask)
    guide = Image.blend(edges, knit_canny(), 0.5)
    out = pipe("a cute dog wearing a thick cable-knit sweater, dog centered full body, plain gray studio background, chunky braided cable knit pattern on sweater",
               image=guide, num_inference_steps=32, guidance_scale=8.0, controlnet_conditioning_scale=1.05,
               negative_prompt="deformed, extra limbs, cropped, embedded in fabric, blurry, watermark").images[0]
    out.save(rf"B:\mode\pet_out6c\pet_c6_{i:02d}.png")
    print("done", i, flush=True)
print("PET_V6C_DONE", flush=True)