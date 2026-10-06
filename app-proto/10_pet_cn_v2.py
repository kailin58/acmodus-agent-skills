# 宠物CN v2:程序合成麻花纹canny线稿(引导图必须含绞花结构)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, math
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image, ImageDraw, ImageFilter
import glob

def knit_canny(size=512):
    """合成针织麻花纹引导图:绞花S辫+罗纹底纹"""
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    # 罗纹竖条(背景织底)
    for x in range(0, size, 24):
        d.line([(x, 0), (x, size)], fill=90, width=3)
    # 3条绞花辫(麻花核心)
    for cx in [size*0.3, size*0.5, size*0.7]:
        pts_top = []
        y = 40
        while y < size - 40:
            amp = 26 * math.sin((y/size) * math.pi)  # 中段幅值大
            d.arc([cx-amp-10, y, cx+amp-10+20, y+34], start=200 if (y//34)%2==0 else 20, end=380, fill=255, width=6)
            y += 34
    im = im.filter(ImageFilter.GaussianBlur(0.8))
    return im

cn = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=cn, torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

os.makedirs("/root/autodl-tmp/pet_out2", exist_ok=True)
srcs = sorted(glob.glob("/root/autodl-tmp/pet_src/*.png"))[:6]
for i, p in enumerate(srcs):
    base = Image.open(p).convert("L").resize((512, 512))
    # 引导=宠物轮廓(轻) + 麻花线稿(强)
    edges = base.filter(ImageFilter.FIND_EDGES).point(lambda v: min(v, 120))
    knit = knit_canny()
    guide = Image.blend(edges, knit, 0.55)
    out = pipe("(a dog wearing a thick cable-knit sweater:1.3), full body dog, studio photo, chunky knit braided pattern",
               image=guide, num_inference_steps=30, guidance_scale=8.0, controlnet_conditioning_scale=1.1,
               negative_prompt="deformed, extra limbs, mutant, blurry, watermark, nude dog").images[0]
    out.save(f"/root/autodl-tmp/pet_out2/pet_cn2_{i:02d}.png")
    print("done", i, flush=True)
print("PET_CN2_DONE", flush=True)
