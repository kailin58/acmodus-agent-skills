# 深度分层脚本 v1.1(1.65m机位版) — 轻量空间模式素材生产工具
# 用法:python depth_layers.py <全景图.jpg> <输出目录>
# 输出:fg.png fg_depth.png mg.png mg_depth.png bg.png bg_depth.png hotspots.json
import sys, os, json
import numpy as np
from PIL import Image, ImageFilter

CAM_HEIGHT_M = 1.65  # 机位高(用户裁定1.55-1.80默认1.65)

def depth_layers(img_path, out_dir):
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    a = np.asarray(im).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    # 1.深度估计(单目启发式:垂直梯度+亮度+饱和度融合)
    gray = a.mean(2)
    # 垂直位置先验(下半=近,上半=远,地面透视)
    vy = np.linspace(1.0, 0.0, H)[:, None] * np.ones((1, W))
    # 局部对比度=细节丰富度(近景细节多)
    sat = a.max(2) - a.min(2) + 1
    detail = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    detail = np.cumsum(detail, axis=0)
    detail = detail / (detail.max() + 1e-6)
    # 深度=加权融合(0-1,1=远)
    depth = 0.55 * (1 - vy) + 0.25 * (1 - detail) + 0.20 * (gray / 255)
    depth = np.clip(depth, 0, 1)
    depth_img = (depth * 255).astype(np.uint8)
    # v1.2优化:对比度拉伸(动态范围扩全)+去噪+边缘保持平滑(中值+轻高斯)
    depth_n = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    dimg = Image.fromarray((depth_n * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3))
    dimg = dimg.filter(ImageFilter.GaussianBlur(2))
    depth_s = np.asarray(dimg).astype(np.float32) / 255
    # 2.三段切分(bg>0.66 / mg 0.33-0.66 / fg<0.33)
    bg_m = depth_s > 0.66
    fg_m = depth_s < 0.33
    mg_m = ~(bg_m | fg_m)
    # 3.层图(每层抠出,其余透明)
    for name, mask in [("fg", fg_m), ("mg", mg_m), ("bg", bg_m)]:
        rgba = np.dstack([a.astype(np.uint8), (mask * 255).astype(np.uint8)])
        Image.fromarray(rgba, "RGBA").save(os.path.join(out_dir, f"{name}.png"))
        # 层深度(该层的深度图,8bit)
        ld = (depth_s * 255).astype(np.uint8)
        Image.fromarray(ld).save(os.path.join(out_dir, f"{name}_depth.png"))
    # 4.热点(简单网格:4角+中心,深度中值排序)
    pts = []
    for cx, cy in [(0.25, 0.6), (0.75, 0.6), (0.5, 0.5), (0.15, 0.5), (0.85, 0.5)]:
        x, y = int(cx * W), int(cy * H)
        d = float(depth_s[y, x])
        pts.append({"u": cx, "v": cy, "depth": round(d, 3),
                    "meters_est": round(1.0 + (1 - d) * 8.0, 2),  # 1-9m粗估
                    "jump_to": None})
    meta = {"camera_height_m": CAM_HEIGHT_M, "width": W, "height": H,
            "method": "heuristic-v1.1(垂直先验+细节+亮度融合)",
            "hotspots": pts}
    json.dump(meta, open(os.path.join(out_dir, "hotspots.json"), "w"), ensure_ascii=False, indent=1)
    print(f"分层完成: {W}x{H} | fg {(fg_m.mean()*100):.0f}% mg {(mg_m.mean()*100):.0f}% bg {(bg_m.mean()*100):.0f}%")
    return out_dir

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else r"B:\mode\models\body3d\lightweight_out"
    if src and os.path.exists(src):
        depth_layers(src, out)
    else:
        # 自测:合成全景(渐变天空+地面)
        W, H = 2048, 1024
        test = np.zeros((H, W, 3), np.uint8)
        test[:H//2] = [135, 180, 220]  # 天
        test[H//2:] = [110, 100, 90]   # 地
        for i in range(H//2, H):
            test[i] = np.clip(test[i] * (1 + (i - H//2) / H), 0, 255)
        tmp = os.path.join(os.environ.get("TEMP", "/tmp"), "test_pano.jpg")
        Image.fromarray(test).save(tmp)
        depth_layers(tmp, out)
        print("自测全景:", tmp)
