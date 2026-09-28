# Pet-Try v32: ControlNet canny引导麻花纹
import math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from peft import PeftModel

# 1) cable canny控制图:在v26图上叠麻花骨架线(只在mask区)
base = Image.open(r"B:\mode\pet\v26_final.png").convert("RGB").resize((512,512))
mask = Image.open(r"B:\mode\pet\v14_mask.png").convert("L")
W, H = 512, 512
ctrl = Image.new("L", (W, H), 0)
d = ImageDraw.Draw(ctrl)
cols = 9
cw = W // cols
# 身体轮廓canny(轻)+麻花列(骨架线)
for c in range(cols):
    cx = c*cw + cw//2
    phase = (c % 2) * math.pi
    pts = []
    for y in range(0, H, 4):
        t = y / H * math.pi * 7 + phase
        off = math.sin(t) * cw * 0.30
        pts.append((cx+off, y))
    d.line(pts, fill=255, width=2)
    # 绞接点横线
    for y in range(0, H, 24):
        d.line((cx-cw*0.3, y, cx+cw*0.3, y+12), fill=180, width=1)
# 只在mask内
m = np.asarray(mask).astype(np.float32)/255.0
ctrl_np = (np.asarray(ctrl).astype(np.float32) * m).astype(np.uint8)
# 混合:暗底图+canny线(ControlNet要白线黑底)
ctrl_img = Image.fromarray(ctrl_np)
bg = Image.new("RGB", (W,H), (0,0,0))
bg.paste(ctrl_img, (0,0))
bg.save(r"B:\mode\pet\v32_canny.png")

# 2) ControlNet img2img
open(r"B:\mode\pet\v32.stage","w").write("cn_loading")
cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
    r"B:\mode\models\sd15\v1-5-pruned-emaonly.safetensors",
    controlnet=cn, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe.enable_model_cpu_offload(); pipe.set_progress_bar_config(disable=True)
pipe.unet = PeftModel.from_pretrained(pipe.unet, r"B:\mode\pet\sd_lora_pet_v22")
g = torch.Generator("cuda").manual_seed(42)
open(r"B:\mode\pet\v32.stage","w").write("generating")
out = pipe("(chunky cable-knit dog sweater:1.3) with twisted rope cables and ribbed texture, covering chest belly and back, studio quality",
           image=base, control_image=bg, strength=0.55, controlnet_conditioning_scale=0.75,
           num_inference_steps=30, guidance_scale=7.0, generator=g,
           negative_prompt="bare fur, naked dog, drool, scarf").images[0]
out.save(r"B:\mode\pet\v32_final.png")
open(r"B:\mode\pet\v32.stage","w").write("done")
print("V32_DONE")
