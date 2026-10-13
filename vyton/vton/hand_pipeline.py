"""vyton.vton.hand_pipeline — 手部修复管线v1(v11b,9.6分)
用法: python hand_pipeline.py <输入图> <输出图>
链路: 指缝检测→条带重绘(SD)→两段放大→自高频纹理→多尺度+光影→边缘锐化80%
依赖: SD1.5本地权重+mediapipe+cv2
"""
import sys, os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import cv2

SD_PATH = r"B:/mode/models/sd15/v1-5-pruned-emaonly.safetensors"

def stage_gap_seek(img):
    """指缝暗线定位(形态学竖线)→条带box列表"""
    g = np.asarray(img.convert("L"))
    H, W = g.shape
    roi = g[:int(H*0.66), :]
    blur = cv2.GaussianBlur(roi, (5,5), 0)
    dark = (blur < np.percentile(blur, 18)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
    lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(lines)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h > 25 and w < 18 and area > 60:
            boxes.append((x-20, y-15, x+w+20, y+h+30))
    return boxes

def stage_gap_redraw(img, boxes):
    """SD条带重绘(clear separation)"""
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_single_file(SD_PATH, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    pipe.enable_attention_slicing()
    W, H = img.size
    out = img.copy()
    for (x0, y0, x1, y1) in boxes:
        mask = Image.new("L", (W, H), 0)
        dm = ImageDraw.Draw(mask)
        dm.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(5))
        g = torch.Generator("cuda").manual_seed(77)
        w2 = 512; h2 = int(H/W*w2)
        out = pipe(
            prompt="macro photo of gap between fingers, clear separation line, dark shadow between fingers, natural skin, photorealistic",
            negative_prompt="fused fingers, webbed skin, blurry, melting",
            image=out.resize((w2, h2)), mask_image=mask.resize((w2, h2)),
            num_inference_steps=24, guidance_scale=7.0, strength=0.85,
            generator=g).images[0].resize((W, H), Image.LANCZOS)
    del pipe; torch.cuda.empty_cache()
    return out

def stage_upscale(img):
    """两段img2img放大512→768→1024"""
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_single_file(SD_PATH, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    pipe.enable_attention_slicing()
    P = "professional photo, hand with manicure, natural skin texture, photorealistic, sharp focus, high detail"
    N = "blurry, melting, plastic, wax, deformed, low quality"
    g1 = torch.Generator("cuda").manual_seed(201)
    mid = pipe(prompt=P, negative_prompt=N, image=img.resize((768, 768), Image.LANCZOS),
               strength=0.25, guidance_scale=6.0, num_inference_steps=22, generator=g1).images[0]
    g2 = torch.Generator("cuda").manual_seed(202)
    hi = pipe(prompt=P, negative_prompt=N, image=mid.resize((1024, 1024), Image.LANCZOS),
              strength=0.22, guidance_scale=6.0, num_inference_steps=24, generator=g2).images[0]
    del pipe; torch.cuda.empty_cache()
    return hi

def stage_texture(img):
    """多尺度毛孔+暗斑减淡+光影gain+噪声统一+锐化80%(v10b+v11b合体,纯CPU)"""
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    H, W = a.shape[:2]
    r, g, b = a[...,0], a[...,1], a[...,2]
    skin_raw = ((r>95)&(g>40)&(b>20)&(r>g)&(r>b)).astype(np.float32)
    skin = cv2.GaussianBlur(skin_raw, (0,0), 8)[..., None]
    lp = cv2.GaussianBlur(a, (0,0), 6); hp = a - lp
    fine = a - cv2.GaussianBlur(a, (0,0), 1.2)
    coarse = cv2.GaussianBlur(a, (0,0), 2.5) - cv2.GaussianBlur(a, (0,0), 5)
    rng = np.random.default_rng(7)
    grain = rng.normal(0, 2.2, (H, W, 1)).astype(np.float32)
    g1 = cv2.GaussianBlur(grain, (0,0), 1.0)[..., None]
    g2 = cv2.GaussianBlur(grain, (0,0), 3.0)[..., None]
    tex = hp*1.35 + fine*0.5 + coarse*0.4 + g1*1.1 + g2*0.5
    o = lp + tex*skin + hp*0.9*(1-skin)
    lum = a.mean(2)
    sk = lum[skin_raw > 0.5]
    if len(sk):
        thr = np.percentile(sk, 25)
        ds = ((lum < thr) & (skin_raw > 0.5)).astype(np.float32)
        ds = cv2.GaussianBlur(ds, (0,0), 6)[..., None]
        o = o * (1 + 0.18*ds*skin)
    hm = float((o*skin).sum()/max(skin.sum()*3,1))
    cm = float((o*(1-skin)).sum()/max((1-skin).sum()*3,1))
    gain = float(np.clip(cm/max(hm,1)*1.03, 0.95, 1.10))
    o = o * (1 + (gain-1)*skin)
    ys, xs = np.where(skin_raw > 0.5)
    if len(ys):
        shadow = np.zeros((H, W), np.float32)
        cv2.ellipse(shadow, (int(xs.mean()), min(H-1, ys.max()+24)), (int(W*0.11), 15), 0, 0, 360, 0.28, -1)
        o = o * (1 - (1-skin)*0.15*shadow[..., None])
    # v11: 甲饰锐化(80%版)+次表面散射+噪声统一
    lumk = o.mean(2, keepdims=True)
    nail = (lumk > np.percentile(lumk, 88)).astype(np.float32)
    nail = cv2.GaussianBlur(nail, (0,0), 3)[..., None]
    sharp = o + (o - cv2.GaussianBlur(o, (0,0), 2.6)) * 0.48
    o = o*(1 - nail*0.5) + sharp*(nail*0.5)
    ss = cv2.GaussianBlur(o*np.array([1.15, 0.9, 0.85]), (0,0), 14)
    o = o*0.88 + ss*0.12*skin
    hi = (lumk > 200).astype(np.float32)
    hi = cv2.GaussianBlur(hi, (0,0), 6)[..., None]
    o = o*(1 - hi*0.10*skin) + (o*0.94)*(hi*0.10*skin)
    rng2 = np.random.default_rng(11)
    uni = cv2.GaussianBlur(rng2.normal(0, 1.4, (H, W, 1)).astype(np.float32), (0,0), 0.8)[..., None]
    o = o + uni*0.8
    return Image.fromarray(np.clip(o, 0, 255).astype(np.uint8))

def repair(src, dst):
    img = Image.open(src).convert("RGB")
    boxes = stage_gap_seek(img)
    print(f"指缝: {len(boxes)}")
    if boxes:
        img = stage_gap_redraw(img, boxes)
    img = stage_upscale(img)
    img = stage_texture(img)
    img.save(dst)
    print("HAND_PIPELINE_DONE", dst)

if __name__ == "__main__":
    repair(sys.argv[1], sys.argv[2])
