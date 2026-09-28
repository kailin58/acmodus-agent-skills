# Pet-Try v33: canny骨架图作为img2img init(SD按线描上色出针织)
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel

ctrl = Image.open(r"B:\mode\pet\v32_canny.png").convert("RGB")
base = Image.open(r"B:\mode\pet\v26_final.png").convert("RGB").resize((512,512))
# 混合:base 60% + ctrl线 40%
from PIL import ImageChops
mix = Image.blend(base, ctrl.point(lambda p: min(p*2,255)).convert("RGB"), 0.35)

pipe = StableDiffusionImg2ImgPipeline.from_single_file(
    r"B:\mode\models\sd15\v1-5-pruned-emaonly.safetensors",
    torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
pipe.unet = PeftModel.from_pretrained(pipe.unet, r"B:\mode\pet\sd_lora_pet_v22")
g = torch.Generator("cuda").manual_seed(7)
out = pipe("photo of a dog wearing (chunky cable-knit sweater:1.3) with twisted rope cables, covering chest belly and back",
           image=mix, strength=0.38, num_inference_steps=30, guidance_scale=7.0, generator=g,
           negative_prompt="bare fur, naked dog, drool, scarf, deformed face").images[0]
out.save(r"B:\mode\pet\v33_final.png")
print("V33_DONE")
