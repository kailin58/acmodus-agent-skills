# B站博主封面采集v2(搜索页直抽,已验证84封面/词)
import subprocess, re, os, sys, time, json, urllib.parse

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUT = r"B:\mode\bz_outfit"
os.makedirs(OUT, exist_ok=True)
names = sys.argv[1].split("|")
IDX = os.path.join(OUT, "bz_index.json")
idx = json.load(open(IDX, encoding="utf-8")) if os.path.exists(IDX) else {}

for nm in names:
    if nm in idx and len(idx[nm]) > 5: continue
    url = f"https://search.bilibili.com/all?keyword={urllib.parse.quote(nm)}"
    r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--user-data-dir=B:\\mode\\pinterest\\chrome_profile",
                        "--window-size=1400,5000",
                        "--dump-dom", "--virtual-time-budget=25000", "--timeout=45000", url],
                       capture_output=True, timeout=90)
    dom = r.stdout.decode("utf-8", errors="replace")
    # 只要archive目录的(视频封面,排除活动图)
    cs = []
    for c in re.findall(r'(i[012]\.hdslb\.com/bfs/archive/[^\s"@\\]{10,90}\.(?:jpg|webp))', dom):
        if c not in cs:
            cs.append(c)
    idx[nm] = cs[:20]
    json.dump(idx, open(IDX, "w", encoding="utf-8"), ensure_ascii=False)
    print(nm, len(cs[:20]), flush=True)
    time.sleep(8)

n = 0
for nm, cs in idx.items():
    for i, c in enumerate(cs):
        dst = os.path.join(OUT, f"{nm[:12]}_{i:02d}.jpg")
        if os.path.exists(dst) and os.path.getsize(dst) > 5000: n += 1; continue
        r = subprocess.run(["curl", "-s", "-o", dst, "-w", "%{http_code}", f"https://{c}"],
                           capture_output=True, text=True, timeout=30)
        if r.stdout.strip() == "200" and os.path.getsize(dst) > 5000: n += 1
        else:
            try: os.remove(dst)
            except: pass
print("BZ_DONE", n, flush=True)
