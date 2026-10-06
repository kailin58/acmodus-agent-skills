# 宠物CN v3:狗轮廓强化(0.75权重)+居中prompt
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, math
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import glob

def knit_canny(size=512):
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    for x in range(0, size, 24):
        d.line([(x, 0), (x, size)], fill=90, width=3)
    for cx in [size*0.3, size*0.5, size*0.7]:
        y = 40
        while y < size - 40:
            amp = 26 * math.sin((y/size) * math.pi)
            d.arc([cx-amp-10, y, cx+amp-10+20, y+34], start=200 if (y//34)%2==0 else 20, end=380, fill=255, width=6)
            y += 34
    return im.filter(ImageFilter.GaussianBlur(0.8))

cn = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=cn, torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

os.makedirs("/root/autodl-tmp/pet_out3", exist_ok=True)
srcs = sorted(glob.glob("/root/autodl-tmp/pet_src/*.png"))[:6]
for i, p in enumerate(srcs):
    base = Image.open(p).convert("L").resize((512, 512))
    # 狗轮廓强化:自动对比度+反色边缘加粗
    base = ImageOps.autocontrast(base, cutoff=8)
    edges = base.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3)).point(lambda v: min(int(v*1.8), 200))
    knit = knit_canny()
    # 狗轮廓主导(0.62),麻花次之
    guide = Image.blend(edges, knit, 0.38)
    out = pipe("a cute dog centered in frame wearing a thick cable-knit sweater covering its torso, full body visible, standing on plain studio floor, braided knit pattern on sweater",
               image=guide, num_inference_steps=32, guidance_scale=8.0, controlnet_conditioning_scale=1.0,
               negative_prompt="deformed, extra limbs, cropped body, dog at edge, embedded in fabric, tapestry, blanket wall, blurry, watermark").images[0]
    out.save(f"/root/autodl-tmp/pet_out3/pet_cn3_{i:02d}.png")
    print("done", i, flush=True)
print("PET_CN3_DONE", flush=True)
