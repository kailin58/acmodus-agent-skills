# IG截图采集法:整页截图→检测格子→切单图(curl被TLS指纹墙,截图绕过)
import subprocess, os, sys, time, urllib.parse
import numpy as np
from PIL import Image

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PROFILE = r"B:\mode\pinterest\chrome_profile"
OUT = r"B:\mode\ig_outfit\cells"
os.makedirs(OUT, exist_ok=True)

TAGS = sys.argv[1].split("|")
for tag in TAGS:
    shot = rf"B:\mode\ig_outfit\shot_{urllib.parse.quote(tag)}.png"
    if os.path.exists(shot) and os.path.getsize(shot) > 200000:
        continue
    url = f"https://www.instagram.com/explore/tags/{urllib.parse.quote(tag)}/"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                    f"--user-data-dir={PROFILE}",
                    "--proxy-server=socks5://127.0.0.1:7897",
                    "--window-size=1400,4400",
                    f"--screenshot={shot}",
                    "--virtual-time-budget=25000", "--timeout=45000", url],
                   capture_output=True, timeout=90)
    time.sleep(7)

# 切格:3列网格
n_total = 0
for f in sorted(os.listdir(r"B:\mode\ig_outfit")):
    if not f.startswith("shot_") or not f.endswith(".png"):
        continue
    im = Image.open(rf"B:\mode\ig_outfit\{f}").convert("RGB")
    a = np.asarray(im)
    lum = a.mean(axis=2)
    rv = lum.std(axis=1)
    rows = np.where(rv > 18)[0]
    if not len(rows):
        continue
    top, bot = int(rows.min()), int(rows.max())
    tag = f[5:-4]
    H = bot - top
    y = top
    k = 0
    while y + 430 <= bot:
        for x in range(0, 1400 - 430, 467):
            cell = im.crop((x, y, x + 430, y + 430))
            ca = np.asarray(cell)
            if ca.std() > 28:  # 有内容才存
                cell.save(rf"{OUT}\{tag}_{n_total:04d}.jpg", quality=92)
                n_total += 1
                k += 1
        y += 460
    print(f"{tag}: {k}格", flush=True)
print("IG_SHOT_DONE", n_total, flush=True)
