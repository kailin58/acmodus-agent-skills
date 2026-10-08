# 画板pins下载器
import json, os, subprocess
IDX = r"B:\mode\pinterest\board_hashes.json"
d = json.load(open(IDX, encoding="utf-8"))
os.makedirs(r"B:\mode\pinterest\board_imgs", exist_ok=True)
n = 0
seen = set()
if os.path.exists(r"B:\mode\pinterest\board_seen.json"):
    seen = set(json.load(open(r"B:\mode\pinterest\board_seen.json")))
for b, hs in d.items():
    for h in hs:
        if h in seen:
            n += 1
            continue
        p3 = "/".join([h[i:i+2] for i in range(0, 6, 2)])
        dst = rf"B:\mode\pinterest\board_imgs\{h}.jpg"
        if os.path.exists(dst) and os.path.getsize(dst) > 5000:
            n += 1
            seen.add(h)
            continue
        r = subprocess.run(["curl", "-s", "-x", "socks5h://127.0.0.1:7897", "-o", dst, "-w", "%{http_code}",
                            f"https://i.pinimg.com/736x/{p3}/{h}.jpg"], capture_output=True, text=True, timeout=40)
        if r.stdout.strip() == "200" and os.path.getsize(dst) > 5000:
            n += 1
            seen.add(h)
        else:
            try: os.remove(dst)
            except: pass
        if n % 20 == 0:
            json.dump(list(seen), open(r"B:\mode\pinterest\board_seen.json", "w"))
print("BOARD_DL_DONE", n)
