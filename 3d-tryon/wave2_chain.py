import subprocess, os
PYUSER = r"C:\Users\Administrator\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages"
LOGDIR = r"B:\mode\plan15h"
TASKS = [
    ("15_pet_v4_sample", r"B:\mode\pet\chain_v20.py", 900),
    ("16_breed_v2d_resume", r"B:\mode\pet\train_breed_v2e.py", 1.5*3600),
    ("17_gender_v5_eval", r"B:\mode\face_data\eval_g5.py", 600),
    ("18_try_v6f_long2k", r"B:\mode\vton_training\distill_v6g.py", 3.5*3600),
    ("19_try_v6g_sample", r"B:\mode\vton_training\sample_v6g.py", 300),
]
for name, script, tmax in TASKS:
    if not os.path.exists(script):
        open(os.path.join(LOGDIR, "_progress.txt"), "a", encoding="utf-8").write(name + ": SKIP(无脚本)\n")
        continue
    log = os.path.join(LOGDIR, name + ".log")
    try:
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.run(["py", "-3.13", "-X", "utf8", script], stdout=f, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONUSERBASE=PYUSER), timeout=int(tmax))
        st = "OK" if p.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired:
        st = "TIMEOUT"
    open(os.path.join(LOGDIR, "_progress.txt"), "a", encoding="utf-8").write(name + ": " + st + "\n")
