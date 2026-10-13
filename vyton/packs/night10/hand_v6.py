# 层1·手部结构重绘管线 v6:21关键点→骨架canny→ControlNet双引导
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

BASE = r"B:\mode\vton_training"
SRC = BASE + "\e2e_hand_v5.png"

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

model_path = r"B:\mode\models\body3d\hand_landmarker.task"
if not os.path.exists(model_path):
    import urllib.request
    urllib.request.urlretrieve("https://hf-mirror.com/qualcomm/MediaPipe/resolve/main/MediaPipe/HandLandmarker/hand_landmarker.task", model_path)

base_opts = mp_python.BaseOptions(model_asset_path=model_path)
opts = mp_vision.HandLandmarkerOptions(base_options=base_opts, num_hands=2, min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3)
det = mp_vision.HandLandmarker.create_from_options(opts)

img = Image.open(SRC).convert("RGB")
W, H = img.size
rgb = np.asarray(img)
mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
res = det.detect(mp_img)
hands = res.hand_landmarks
if not hands:
    # ROI放大二次检测(手太小/图大):上放大2倍再检,坐标缩回
    big = img.resize((W*4, H*4), Image.LANCZOS)
    # 加白边(手贴图边检测难)
    W4, H4 = big.size
    padded = Image.new("RGB", (W4+200, H4+200), (255,255,255))
    padded.paste(big, (100, 100))
    big = padded
    rgb2 = np.asarray(big)
    res2 = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb2))
    class _H:
        pass
    for h2 in res2.hand_landmarks:
        class _P:
            pass
        h = _H()
        h.landmark = []
        W4p, H4p = big.size
        for p in h2:
            q = _P()
            q.x = (p.x*W4p - 100) / W4
            q.y = (p.y*H4p - 100) / H4
            q.z = p.z
            h.landmark.append(q)
        hands.append(h)
    print(f"二次检测(放大2x): {len(hands)}", flush=True)
else:
    print(f"检测到手: {len(hands)}", flush=True)

BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
         (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
canvas = Image.new("L", (W, H), 0)
d = ImageDraw.Draw(canvas)
if hands:
    for h in hands:
        lms = h if isinstance(h, list) else h.landmark
        pts = [(p.x*W, p.y*H) for p in lms]
        for a, b in BONES:
            d.line([pts[a], pts[b]], fill=255, width=3)
        for p in pts:
            d.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill=255)
if not hands:
    # 检测失败→原图canny边缘引导(保结构不依赖手检测器)
    import cv2
    g = cv2.Canny(np.asarray(img.convert("L")), 60, 150)
    canvas = Image.fromarray(g)
    print("手检测0,退化canny原图引导", flush=True)
else:
    canvas = canvas.filter(ImageFilter.GaussianBlur(1.2))
canvas.save(BASE + "\hand_v6_canny.png")
print("control图完成", flush=True)

if hands:
    xs = [p.x*W for h in hands for p in (h if isinstance(h, list) else h.landmark)]
    ys = [p.y*H for h in hands for p in (h if isinstance(h, list) else h.landmark)]
    x0, x1 = max(0, int(min(xs))-30), min(W, int(max(xs))+30)
    y0, y1 = max(0, int(min(ys))-30), min(H, int(max(ys))+30)
else:
    x0, y0, x1, y1 = int(W*0.2), 0, int(W*0.8), int(H*0.6)
mask = Image.new("L", (W, H), 0)
dm = ImageDraw.Draw(mask)
dm.rounded_rectangle([x0, y0, x1, y1], radius=24, fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(8))
mask.save(BASE + "\hand_v6_mask.png")
print(f"手部box: {x0},{y0},{x1},{y1}", flush=True)

from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline

cn = ControlNetModel.from_single_file(
    r"B:\mode\models\control_v11p_sd15_canny.pth",
    torch_dtype=torch.float16)
pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
    r"B:\mode\models\sd15\v1-5-pruned-emaonly.safetensors",
    controlnet=cn, torch_dtype=torch.float16, safety_checker=None)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

g = torch.Generator("cuda").manual_seed(123)
w2 = 512
h2 = int(H/W*w2)
out = pipe(
    prompt="professional photo of a hand with five distinct fingers, natural skin texture with pores, detailed knuckles, clear finger gaps, french manicure nails, consistent lighting, photorealistic, 8k",
    negative_prompt="fused fingers, extra fingers, missing fingers, deformed hand, blurry, melting, plastic, wax, cartoon",
    image=img.resize((w2, h2)),
    control_image=canvas.resize((w2, h2)),
    mask_image=mask.resize((w2, h2)),
    num_inference_steps=28,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.85,
    generator=g,
).images[0]
out = out.resize((W, H), Image.LANCZOS)
out.save(BASE + "\e2e_hand_v6_1.png")
print("V6_DONE", flush=True)
