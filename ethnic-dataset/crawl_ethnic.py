# Wikimedia Commons传统服饰采集(公版,API礼貌限速)
import json, os, subprocess, time, re

COUNTRIES = ["Japan", "China", "South_Korea", "Vietnam", "Thailand", "Indonesia",
             "India", "Pakistan", "Saudi_Arabia", "Iran", "Turkey", "Russia",
             "France", "Germany", "Italy", "Egypt", "South_Africa", "Brazil", "Mexico", "Nigeria"]
OUT = r"B:\mode\ethnic_data"
os.makedirs(OUT, exist_ok=True)
report = {}
HDR = ["-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"]
for c in COUNTRIES:
    cdir = os.path.join(OUT, c)
    os.makedirs(cdir, exist_ok=True)
    cat = f"Category:Traditional clothing of {c.replace('_', ' ')}"
    # 成员列表
    r = subprocess.run(["curl", "-sL", "-x", "socks5h://127.0.0.1:7897", "-m", "25"] + HDR + [
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers&gcmtitle={cat.replace(' ', '_')}&gcmtype=file&gcmlimit=60&prop=imageinfo&iiprop=url|size&iiurlwidth=800&format=json"],
        capture_output=True, text=True, timeout=40)
    try:
        d = json.loads(r.stdout or "{}")
        pages = d.get("query", {}).get("pages", {})
    except Exception:
        pages = {}
    n = 0
    for pid, p in pages.items():
        if n >= 25: break
        ii = (p.get("imageinfo") or [{}])[0]
        u = ii.get("thumburl") or ii.get("url")
        if not u: continue
        title = p.get("title", "").replace("File:", "").split(".")[0][:40]
        ext = u.split("?")[0].split(".")[-1][:4]
        dst = os.path.join(cdir, f"{title}_{n:02d}.{ext}".replace(" ", "_"))
        try:
            rd = subprocess.run(["curl", "-sL", "-x", "socks5h://127.0.0.1:7897", "-m", "20"] + HDR + ["-o", dst, u],
                                capture_output=True, timeout=30)
            if os.path.exists(dst) and os.path.getsize(dst) > 40000:
                n += 1
            elif os.path.exists(dst):
                os.remove(dst)
        except Exception:
            pass
        time.sleep(0.8)  # 礼貌限速
    report[c] = n
    print(c, n, flush=True)
    time.sleep(1.5)
json.dump(report, open(os.path.join(OUT, "_report.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("ETHNIC_DONE", sum(report.values()), flush=True)
