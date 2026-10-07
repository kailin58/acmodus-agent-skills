# IG标签采集:headless dump -> scontent图链 -> 下载
import subprocess, re, json, os, sys, time, urllib.parse

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PROFILE = r"B:\mode\pinterest\chrome_profile"
OUT_DIR = r"B:\mode\ig_outfit"
IDX = os.path.join(OUT_DIR, "ig_index.json")
os.makedirs(OUT_DIR, exist_ok=True)

TAGS = sys.argv[1].split("|")
idx = json.load(open(IDX, encoding="utf-8")) if os.path.exists(IDX) else {}

for tag in TAGS:
    if tag in idx:
        continue
    url = f"https://www.instagram.com/explore/tags/{urllib.parse.quote(tag)}/"
    try:
        r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                            f"--user-data-dir={PROFILE}",
                            "--proxy-server=socks5://127.0.0.1:7897",
                            "--window-size=1400,3000",
                            "--dump-dom", "--virtual-time-budget=20000", "--timeout=40000", url],
                           capture_output=True, timeout=90)
        dom = r.stdout.decode("utf-8", errors="replace")
        links = sorted(set(re.findall(r'(scontent[^"\s]{20,250}\.(?:jpg|webp))', dom)))
        idx[tag] = links
        json.dump(idx, open(IDX, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"{tag}: {len(links)}图", flush=True)
    except Exception as e:
        print(f"{tag}: ERR {str(e)[:40]}", flush=True)
    time.sleep(8)
print("IG_TAGS_DONE", flush=True)
