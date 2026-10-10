# 宠物v8:vision教训修正——全身毛衣提示词强化+负提示强化织物吞没
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, math, glob, json
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

srcs = sorted(glob.glob(r"B:\mode\pet\lora_data_v3\dog_*_src.png"))[:3]
os.makedirs(r"B:\mode\pet_out8", exist_ok=True)
gen = torch.Generator("cuda")
n = 0
# vision修正:full body强提示+负面加"head only swallowed by fabric"
PROMPT = ("a cute dog standing full body wearing a thick cable-knit sweater covering its torso and four legs, "
          "dog fully visible head torso front legs back legs tail, sweater fits body like real clothing with sleeves, "
          "plain gray studio background, chunky braided cable knit pattern")
NEG = ("deformed, extra limbs, cropped, embedded in fabric, head swallowed by fabric, fabric covering face, "
       "knit mask over muzzle, blanket instead of clothing, dog merging into textile, blurry, watermark, "
       "missing legs, missing body, dissolving, texture tiled flat")
for i, p in enumerate(srcs):
    im0 = Image.open(p).convert("L").resize((456, 440))
    canvas = Image.new("L", (512, 512), 255)
    canvas.paste(im0, (28, 46))
    base = ImageOps.autocontrast(canvas, cutoff=8)
    edges = base.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3)).point(lambda v: min(int(v*1.8), 200))
    Hmask = Image.new("L", (512, 512), 60)
    d2 = ImageDraw.Draw(Hmask)
    d2.ellipse([150, 30, 380, 250], fill=200)
    edges = Image.composite(edges, edges.point(lambda v: int(v*0.4)), Hmask)
    guide = Image.blend(edges, knit_canny(), 0.5)
    for s in [123, 77]:
        for w in [0.9, 1.1]:
            gen.manual_seed(s + i*100)
            out = pipe(PROMPT, image=guide, num_inference_steps=32, guidance_scale=8.0,
                       controlnet_conditioning_scale=w, negative_prompt=NEG, generator=gen).images[0]
            out.save(rf"B:\mode\pet_out8\v8_{n:02d}_s{s}_c{w}.png")
            n += 1
    print("done", i, flush=True)
print("PET_V8_DONE", n, flush=True)
