# VytonSR-R2数据生成:500对(ETC1S解码lo, 原图hi)
import subprocess, os, glob, random, shutil
from PIL import Image

KTX = r"C:/Program Files/KTX-Software/bin/ktx.exe"
ENV = dict(os.environ, PATH=os.environ.get("PATH","") + ";C:/Program Files/KTX-Software/bin")
SRC = r"B:/mode/models/sr/pairs"

random.seed(1)
pool = []
for pat in [r"B:/mode/pinterest/outfit_imgs/*.jpg",
            r"B:/mode/pinterest/board_imgs/*.jpg",
            r"B:/mode/yt_outfit/*.jpg"]:
    pool += glob.glob(pat)
random.shuffle(pool)

n = 0
for p in pool:
    if n >= 500: break
    try:
        im = Image.open(p).convert("RGB")
        im.thumbnail((512, 512))
        if im.size[0] < 300: continue
        hi_p = f"{SRC}/hi/{n:04d}.png"
        lo_tmp = f"{SRC}/lo/{n:04d}_c.ktx2"
        im.save(hi_p)
        r = subprocess.run([KTX, "create", "--format", "R8G8B8A8_UNORM", "--encode", "basis-lz", hi_p, lo_tmp],
                           capture_output=True, timeout=60, env=ENV)
        if not os.path.exists(lo_tmp): continue
        r2 = subprocess.run([KTX, "extract", lo_tmp, f"{SRC}/lo/{n:04d}_d", "--all"],
                            capture_output=True, timeout=60, env=ENV)
        dec = glob.glob(f"{SRC}/lo/{n:04d}_d*/output.png") or glob.glob(f"{SRC}/lo/{n:04d}_d*.png")
        if not dec: continue
        shutil.move(dec[0], f"{SRC}/lo/{n:04d}.png")
        os.remove(lo_tmp)
        for d in glob.glob(f"{SRC}/lo/{n:04d}_d*"):
            shutil.rmtree(d, ignore_errors=True)
        n += 1
        if n % 100 == 0: print(n, flush=True)
    except Exception:
        continue
print(f"R2_PAIRS_DONE {n}")
