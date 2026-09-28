import subprocess, os, time
PYUSER = r"C:\Users\Administrator\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages"
LOGDIR = r"B:\mode\plan15h"
TASKS = [
    ("11_breed_v2d_fix", r"B:\mode\pet\train_breed_v2d.py", 1.8*3600),
    ("12_style_v4g_fix", r"B:\mode\style_train\train_v4g.py", 3600),
]
for name, script, tmax in TASKS:
    log = os.path.join(LOGDIR, name + ".log")
    try:
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.run(["py", "-3.13", "-X", "utf8", script], stdout=f, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONUSERBASE=PYUSER), timeout=int(tmax))
        st = "OK" if p.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired:
        st = "TIMEOUT"
    open(os.path.join(LOGDIR, "_progress.txt"), "a", encoding="utf-8").write(name + ": " + st + "\n")
