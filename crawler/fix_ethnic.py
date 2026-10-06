import json, os, subprocess, time
FIX = {"Egypt": "Egyptian_clothing", "Iran": "Clothing_of_Iran", "Saudi_Arabia": "Saudi_clothing", "South_Korea": "Hanbok", "Pakistan": "Pakistani_clothing"}
OUT = r"B:\mode\ethnic_data"
for c, cat in FIX.items():
    cdir = os.path.join(OUT, c)
    os.makedirs(cdir, exist_ok=True)
    url = "https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers&gcmtitle=Category:" + cat + "&gcmtype=file&gcmlimit=40&prop=imageinfo&iiprop=url&iiurlwidth=800&format=json"
    r = subprocess.run(["curl", "-sL", "-x", "socks5h://127.0.0.1:7897", "-m", "25", url], capture_output=True, text=True, timeout=40)
    try:
        d = json.loads(r.stdout or "{}")
        pages = d.get("query", {}).get("pages", {})
    except Exception:
        pages = {}
    n = len([f for f in os.listdir(cdir) if not f.startswith("_")])
    for pid, p in pages.items():
        if n >= 25: break
        ii = (p.get("imageinfo") or [{}])[0]
        u = ii.get("thumburl") or ii.get("url")
        if not u: continue
        ext = u.split("?")[0].split(".")[-1][:4]
        dst = os.path.join(cdir, "x_%02d.%s" % (n, ext))
        try:
            subprocess.run(["curl", "-sL", "-x", "socks5h://127.0.0.1:7897", "-m", "20", "-o", dst, u], timeout=30)
            if os.path.exists(dst) and os.path.getsize(dst) > 40000: n += 1
            elif os.path.exists(dst): os.remove(dst)
        except Exception: pass
        time.sleep(0.8)
    print(c, n, flush=True)
print("FIX_DONE")