
# 脸部修复:检测脸→裁剪放大→img2img重绘→贴回(v1,通用型,不依赖GFPGAN权重)
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel
from facemesh_lib.tasks import python as mp_python
from facemesh_lib.tasks.python import vision as mp_vision
from facemesh_lib import Image as MPImage, ImageFormat

src = Image.open(r"B:\mode\vton_training\samples\v5_0.png").convert("RGB")
W, H = src.size
# 脸太糊landmarker检测不到→用启发式:上半图中央区域(全身照脸部典型位置)
cx, cy = W*0.5, H*0.16
size = H*0.28
x0, y0 = int(max(cx-size/2,0)), int(max(cy-size/2,0))
x1, y1 = int(min(cx+size/2,W)), int(min(cy+size/2,H))
face = src.crop((x0,y0,x1,y1)).resize((512,512))

pipe = StableDiffusionImg2ImgPipeline.from_single_file(
    r"B:\mode\models\sd15\v1-5-pruned-emaonly.safetensors",
    torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
fixed = pipe("a clear detailed human face, sharp eyes, symmetrical features, fashion portrait",
             image=face, strength=0.32, num_inference_steps=24,
             guidance_scale=6.5).images[0]
fixed = fixed.resize((x1-x0, y1-y0))
out = src.copy(); out.paste(fixed, (x0, y0))
out.save(r"B:\mode\vton_training\samples\v5_0_faceref.png")
print("FACE_REF_DONE")
