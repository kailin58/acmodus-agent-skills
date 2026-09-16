# -*- coding: utf-8 -*-
"""Pinterest 公开搜索页采集:headless Chrome dump-dom -> 提取pin图hash
支持断点续跑(已有关键词跳过),输出 JSON 索引。"""
import subprocess, urllib.parse, re, json, os, sys, time

CHROME = r"C:\Users\Administrator\.agent-browser\browsers\chrome-153.0.8010.47\chrome.exe"
BASE = r"B:\mode\pinterest"
HTML_DIR = os.path.join(BASE, "html")
os.makedirs(HTML_DIR, exist_ok=True)

KEYWORDS = sys.argv[1].split("|") if len(sys.argv) > 1 else []
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "hashes.json")

hashes_by_kw = {}
if os.path.exists(OUT):
    hashes_by_kw = json.load(open(OUT, encoding="utf-8"))

for kw in KEYWORDS:
    if kw in hashes_by_kw and hashes_by_kw[kw]:
        print(f"skip {kw}", flush=True)
        continue
    url = "https://www.pinterest.com/search/pins/?q=" + urllib.parse.quote(kw) + "&rs=typed"
    try:
        r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                            "--dump-dom", "--timeout=35000", "--virtual-time-budget=10000", url],
                           capture_output=True, timeout=70)
        html = r.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        html = ""
    safe = re.sub(r"[^\w\u4e00-\u9fff]+", "_", kw)
    open(os.path.join(HTML_DIR, safe + ".html"), "w", encoding="utf-8").write(html)
    hs = list(dict.fromkeys(re.findall(r'i\.pinimg\.com/\d+x(?:/[0-9a-f]{2}){3}/([0-9a-f]{24,40})\.', html)))
    hashes_by_kw[kw] = hs
    json.dump(hashes_by_kw, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{kw}: {len(hs)} pins ({len(html)//1024}KB)", flush=True)
    time.sleep(1)

allh = {}
for kw, v in hashes_by_kw.items():
    for h in v:
        allh.setdefault(h, []).append(kw)
print("TOTAL unique:", len(allh), flush=True)
