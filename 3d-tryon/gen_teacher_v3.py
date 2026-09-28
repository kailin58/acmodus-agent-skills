# 教师对扩产:VITON人图→hd2管线(txt2img+LoRA v5)→穿衣教师图
import io, os, glob
import pyarrow.parquet as pq
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

OUT = r"B:\mode\vton_training\teacher_pairs_v3"
os.makedirs(OUT, exist_ok=True)
# 取VITON人图+对应garment caption
pf = pq.ParquetFile(r"B:\mode\vton_training\data\VITON-HD-Captions\data\train-00000-of-00012.parquet")
t = pf.read()
rows = []
for i in range(600, min(2300, t.num_rows)):
    row = {c: t.column(c)[i].as_py() for c in t.column_names}
    rows.append(row)
print("VITON行:", len(rows), flush=True)

pipe = StableDiffusionPipeline.from_single_file(
    r"B:\mode\models\sd15\v1-5-pruned-emaonly.safetensors",
    torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
pipe.unet = PeftModel.from_pretrained(pipe.unet, r"B:\mode\vton_training\sd_lora_v5")

n = 0
for k, row in enumerate(rows):
    cap = str(row.get("text", ""))[:120]
    if not cap: continue
    try:
        g = torch.Generator("cuda").manual_seed(2000+k)
        out = pipe(cap + ", fashion model wearing it, studio lighting, full body, head visible",
                   num_inference_steps=24, guidance_scale=6.5, generator=g,
                   negative_prompt="cropped head, blurry, low quality, deformed").images[0]
        out.save(os.path.join(OUT, f"t_{k:04d}.png"))
        n += 1
        if n % 20 == 0: print("产", n, flush=True)
    except Exception as e:
        print("skip", k, str(e)[:50], flush=True)
        continue
print("TEACHER_V3_DONE", n, flush=True)
