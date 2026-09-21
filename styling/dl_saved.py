
import subprocess, os, json, hashlib
imgs = json.load(open(r"B:\mode\pinterest\saved_pins.json"))
dst = r"B:\mode\pinterest\images"
existing = {f for f in os.listdir(dst)}
ok = 0
for i, u in enumerate(imgs):
    big = u.replace("/236x/", "/736x/")
    h = hashlib.md5(u.encode()).hexdigest()[:12]
    fn = f"saved_{h}.jpg"
    if fn in existing: ok += 1; continue
    subprocess.run(["curl", "-s", "-m", "20", "-x", "socks5h://127.0.0.1:7897",
                    "-A", "Mozilla/5.0 Chrome/120.0", "-o", os.path.join(dst, fn), big],
                   capture_output=True, timeout=30)
    if os.path.exists(os.path.join(dst, fn)) and os.path.getsize(os.path.join(dst, fn)) > 8000:
        ok += 1
print("DONE", ok, "/", len(imgs))
