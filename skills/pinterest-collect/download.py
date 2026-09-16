# -*- coding: utf-8 -*-
"""从 pin hash 下载图片(经代理,originals 优先,失败降级 736x)。断点续跑。"""
import json, os, subprocess, sys, time

BASE = r"B:\mode\pinterest"
IMG = os.path.join(BASE, "images")
os.makedirs(IMG, exist_ok=True)
allh = json.load(open(os.path.join(BASE, "all_pins.json"), encoding="utf-8"))

# hash -> 目录路径映射: pinimg originals 布局 = / originals / ab/cd/hash.ext
# 需要子目录,从 html 里抓完整路径
import re, glob
path_map = {}
for f in glob.glob(os.path.join(BASE, "html", "*.html")):
    html = open(f, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r'i\.pinimg\.com/236x(/[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{2}/([0-9a-f]{24,40}))\.', html):
        path_map[m.group(2)] = m.group(1)

print("path_map:", len(path_map), "/", len(allh))
ok = fail = skip = 0
for i, (h, kws) in enumerate(allh.items()):
    dest = os.path.join(IMG, h + ".jpg")
    if os.path.exists(dest) and os.path.getsize(dest) > 5000:
        skip += 1; continue
    if h not in path_map:
        fail += 1; continue
    p = path_map[h]
    urls = [f"https://i.pinimg.com/originals{p}.jpg",
            f"https://i.pinimg.com/736x{p}.jpg"]
    got = False
    for u in urls:
        r = subprocess.run(["curl", "-s", "-m", "25", "--proxy", "socks5h://127.0.0.1:7897",
                            "-o", dest, "-w", "%{http_code}", u], capture_output=True, text=True)
        if r.stdout.strip() == "200" and os.path.exists(dest) and os.path.getsize(dest) > 5000:
            got = True; break
    if got: ok += 1
    else:
        fail += 1
        if os.path.exists(dest): os.remove(dest)
    if (i+1) % 10 == 0:
        print(f"{i+1}/{len(allh)} ok={ok} fail={fail} skip={skip}", flush=True)
    time.sleep(0.4)
print(f"DONE ok={ok} fail={fail} skip={skip}")
