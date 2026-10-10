# YouTube穿搭/化妆缩略图采集+博主频道库
import subprocess, re, os, sys, time, json, urllib.parse

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUT = r"B:\mode\yt_outfit"
os.makedirs(OUT, exist_ok=True)
KWS = sys.argv[1].split("|")

IDX = os.path.join(OUT, "yt_index.json")
idx = json.load(open(IDX, encoding="utf-8")) if os.path.exists(IDX) else {}

for kw in KWS:
    if kw in idx: continue
    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(kw)}"
    r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--user-data-dir=B:\\mode\\pinterest\\chrome_profile",
                        "--proxy-server=socks5://127.0.0.1:7897",
                        "--window-size=1400,4000",
                        "--dump-dom", "--virtual-time-budget=20000", "--timeout=40000", url],
                       capture_output=True, timeout=80)
    dom = r.stdout.decode("utf-8", errors="replace")
    vids = list(dict.fromkeys(re.findall(r'/watch\?v=([-\w]{11})', dom)))[:30]
    chans = re.findall(r'"ownerText":\{"runs":\[\{"text":"([^"]+)"', dom)
    idx[kw] = {"videos": vids, "channels": list(dict.fromkeys(chans))[:10]}
    json.dump(idx, open(IDX, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{kw}: {len(vids)}v {len(chans)}ch", flush=True)
    time.sleep(6)

n = 0
for kw, d in idx.items():
    for vid in d["videos"]:
        dst = os.path.join(OUT, f"{vid}.jpg")
        if os.path.exists(dst) and os.path.getsize(dst) > 5000:
            n += 1; continue
        for q in ["maxresdefault", "hqdefault"]:
            r = subprocess.run(["curl", "-s", "-x", "socks5h://127.0.0.1:7897", "-o", dst, "-w", "%{http_code}",
                                f"https://i.ytimg.com/vi/{vid}/{q}.jpg"], capture_output=True, text=True, timeout=30)
            if r.stdout.strip() == "200" and os.path.getsize(dst) > 5000:
                n += 1; break
            else:
                try: os.remove(dst)
                except: pass
    print("dl:", n, flush=True)
print("YT_DONE", n, flush=True)