
# v7第一步:指缝暗线定位(中指-无名指间隙黑线=v6.1vision点名的粘连)
import numpy as np
from PIL import Image
import cv2
img = np.asarray(Image.open(r"B:\mode\vton_training\e2e_hand_v6_1.png").convert("RGB"))
H, W = img.shape[:2]
g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# 手部区域(上2/3)
roi = g[:int(H*0.66), :]
# 暗细线检测:局部极小值+纵向连续
blur = cv2.GaussianBlur(roi, (5,5), 0)
dark = (blur < np.percentile(blur, 18)).astype(np.uint8)
# 形态学:竖向细长的暗结构
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
n, lbl, stats, cent = cv2.connectedComponentsWithStats(lines)
cands = []
for i in range(1, n):
    x, y, w2, h2, area = stats[i]
    if h2 > 25 and w2 < 18 and area > 60:  # 细长竖暗线=指缝候选
        cands.append((x, y, w2, h2, area))
print("指缝候选:", len(cands))
for c in cands[:8]:
    print("  x%d y%d w%d h%d a%d" % c)
# 可视化
vis = img.copy()
for x, y, w2, h2, a in cands[:8]:
    cv2.rectangle(vis, (x, y), (x+w2, y+h2), (255, 0, 0), 2)
Image.fromarray(vis[:int(H*0.66), :]).save(r"B:\mode\vton_training\hand_v7_gapseek.png")
print("GAPSEEK_OK")
