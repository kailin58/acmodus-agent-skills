# -*- coding: utf-8 -*-
# 官网采集:第一批30家,每家限20张礼貌下载
import json, os, re, sys, time
sys.path.insert(0, r"B:\mode")
from polite_crawler import PoliteCrawler

BATCH = json.load(open(r"B:\mode\crawl_batch1.json", encoding="utf-8"))
OUT = r"B:\mode\brand_site"
os.makedirs(OUT, exist_ok=True)
report = []
import glob as _g
_done = set()
for _d in _g.glob(os.path.join(OUT, "*")):
    if os.path.isdir(_d) and len(_g.glob(os.path.join(_d, "img_*"))) >= 15:
        _done.add(os.path.basename(_d))
for brand, url, _code in BATCH:
    bkey = brand.replace(" ", "_").replace("/", "-")[:40]
    if bkey in _done:
        print(brand, "SKIP已采", flush=True)
        continue
    bdir = os.path.join(OUT, brand.replace(" ", "_").replace("/", "-")[:40])
    os.makedirs(bdir, exist_ok=True)
    try:
        import threading
        # per-brand watchdog: 家级3分钟上限
        _stop = [False]
        def _wd():
            time.sleep(180)
            _stop[0] = True
        threading.Thread(target=_wd, daemon=True).start()
        dom = url.split("//")[1].split("/")[0]
        c = PoliteCrawler(dom)
        html = c.fetch(url)
        if not isinstance(html, str) or len(html) < 500:
            report.append([brand, "FAIL首页"]); continue
        imgs = re.findall(r'(?:src|data-src)=["\'' + chr(39) + ']([^"\'' + chr(39) + ']+\.(?:jpg|jpeg|png|webp))["\'' + chr(39) + ']', html, re.I)
        imgs += re.findall(r'url\(["\'' + chr(39) + ']?([^"\')]+\.(?:jpg|jpeg|png|webp))', html, re.I)
        seen, n_ok = set(), 0
        for u in imgs:
            if n_ok >= 20: break
            if u.startswith("data:"): continue
            if u.startswith("//"): u = "https:" + u
            elif u.startswith("/"): u = "https://" + dom + u
            elif not u.startswith("http"): continue
            if any(x in u.lower() for x in ["logo", "icon", "sprite", "payment", "flag", ".svg"]): continue
            if u in seen: continue
            seen.add(u)
            ext = u.split("?")[0].split(".")[-1][:4]
            dst = os.path.join(bdir, "img_%03d.%s" % (n_ok, ext))
            try:
                c.fetch(u, out=dst)
                if os.path.getsize(dst) > 30000:
                    n_ok += 1
                else:
                    if os.path.exists(dst): os.remove(dst)
            except Exception:
                pass
        report.append([brand, n_ok])
        print(brand, n_ok, flush=True)
    except Exception as e:
        report.append([brand, "ERR:" + str(e)[:40]])
json.dump(report, open(os.path.join(OUT, "_batch1_report.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("CRAWL_B1_DONE", sum(r[1] for r in report if isinstance(r[1], int)), flush=True)
