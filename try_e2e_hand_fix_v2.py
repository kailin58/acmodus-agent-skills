
# Try e2e手部修复v2:SD inpaint手部区域(手指重生成)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, numpy as np
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler

pipe = StableDiffusionInpaintPipeline.from_single_file(
    r"B:\mode\models\sd15_pruned_emaonly.safetensors",
    torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

img = Image.open(r"B:\mode\vton_training\e2e_out_0.png").convert("RGB")
W, H = img.size
# 手部mask(上1/3的皮肤区,刚才定位的bbox扩一点)
mask = Image.new("L", (W, H), 0)
d = ImageDraw.Draw(mask)
d.ellipse([70, 10, 310, 200], fill=255)  # 举起的手+发交界
gen = torch.Generator("cuda").manual_seed(123)
out = pipe("a woman's natural raised left hand with five distinct fingers, fingers clearly separated, realistic skin texture, natural lighting, photorealistic",
           image=img.resize((512, 512)), mask_image=mask.resize((512, 512)),
           num_inference_steps=28, guidance_scale=7.5,
           negative_prompt="deformed hand, fused fingers, extra fingers, blurry hand, melted fingers, watermark",
           generator=gen).images[0]
out = out.resize((W, H), Image.LANCZOS)
out.save(r"B:\mode\vton_training\e2e_hand_inpaint.png")
print("INPAINT_OK")
