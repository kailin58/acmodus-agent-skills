# -*- coding: utf-8 -*-
import subprocess, os, glob, time, json

PYUSER = r"C:\Users\Administrator\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages"
def run(name, script, tmax):
    log = os.path.join(r"B:\mode\plan15h", name + ".log")
    try:
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.run(["py", "-3.13", "-X", "utf8", "-u", script], stdout=f, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONUSERBASE=PYUSER), timeout=int(tmax))
        return "OK" if p.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"

def n_imgs():
    return len(glob.glob(r"B:\mode\brand_site\*\img_*.*"))

for _ in range(120):
    lg = open(r"B:\mode\plan15h\71f.log", encoding="utf-8", errors="replace").read() if os.path.exists(r"B:\mode\plan15h\71f.log") else ""
    if "V12R_DONE" in lg:
        run("72f", r"B:\mode\vton_training\sample_v12r.py", 400)
        break
    time.sleep(60)

for _ in range(300):
    if n_imgs() >= 150:
        break
    time.sleep(60)
run("81_style_v5", r"B:\mode\style_train\train_v5.py", 5400)

rep = r"B:\mode\brand_site\_batch1_report.json"
if os.path.exists(rep):
    ok = json.load(open(r"B:\mode\brand_url_check.json", encoding="utf-8"))["ok"]
    b2 = ok[30:80]
    json.dump(b2, open(r"B:\mode\crawl_batch1.json", "w", encoding="utf-8"), ensure_ascii=False)
    run("82_crawl_b2", r"B:\mode\run_crawl.py", 14400)
print("AUTO_SCHED_DONE", flush=True)