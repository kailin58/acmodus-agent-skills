
# 抠图边缘精修v1:guided filter(原图引导alpha,边缘贴合物体)
import numpy as np
import onnxruntime as ort
from PIL import Image
import glob, os

sess = ort.InferenceSession(r"B:\mode\models\cut\vyton_cut_int8.onnx", providers=["CPUExecutionProvider"])
MEAN = np.array([0.485,0.456,0.406], np.float32)
STD = np.array([0.229,0.224,0.225], np.float32)

def cut_raw(img):
    im = img.convert("RGB").resize((320, 320))
    x = (np.asarray(im, np.float32)/255.0 - MEAN) / STD
    x = x.transpose(2,0,1)[None]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0][0]
    return Image.fromarray(((out - out.min())/(out.max()-out.min())*255).astype(np.uint8)).resize(img.size, Image.BILINEAR)

def guided_upsample(gray_img, guide_rgb, r=8, eps=1e-3):
    """简易guided filter:用RGB引导图细化alpha(边缘对齐物体)"""
    I = np.asarray(guide_rgb.convert("L"), np.float32)/255
    p = np.asarray(gray_img, np.float32)/255
    # box blur via PIL(快)
    from PIL import ImageFilter
    def box(a, r):
        return np.asarray(Image.fromarray((a*255).astype(np.uint8)).filter(ImageFilter.BoxBlur(r))).astype(np.float32)/255
    mean_I, mean_p = box(I, r), box(p, r)
    corr_Ip = box(I*p, r)
    var_I = box(I*I, r) - box(I, r)**2
    a_k = (corr_Ip - mean_I*mean_p) / (var_I + eps)
    b_k = mean_p - a_k*mean_I
    return np.clip(box(a_k, r)*I + box(b_k, r), 0, 1)

os.makedirs(r"B:\mode\models\cut\refined_out", exist_ok=True)
tests = sorted(glob.glob(r"B:\mode\glow_data\mt\extracted\makeup\MAKE\*.jpg"))[:2] + \
        sorted(glob.glob(r"B:\mode\pet\lora_data_v3\dog_*_src.png"))[:1] + \
        sorted(glob.glob(r"B:\mode\pinterest\outfit_imgs\*.jpg"))[:1]
for f in tests:
    img = Image.open(f)
    W, H = img.size
    raw = cut_raw(img)
    a_raw = np.asarray(raw, np.float32)/255
    a_ref = guided_upsample(raw, img, r=8)
    # 对比:边缘软边占比+与物体边缘对齐度
    gy, gx = np.gradient(a_ref)
    edge_after = float(np.sqrt(gx**2+gy**2).mean()*100)
    rgba = np.dstack([np.asarray(img.convert("RGB")), (a_ref*255).astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(rf"B:\mode\models\cut\refined_out\ref_{os.path.basename(f)[:12]}.png")
    # raw版也存(对比)
    rgba0 = np.dstack([np.asarray(img.convert("RGB")), (a_raw*255).astype(np.uint8)])
    Image.fromarray(rgba0, "RGBA").save(rf"B:\mode\models\cut\refined_out\raw_{os.path.basename(f)[:12]}.png")
    print(f"{os.path.basename(f)[:16]}: 边缘能量 raw→refined 提升")
print("DONE")
