import subprocess, os
PYUSER = r"C:\Users\Administrator\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages"
LOGDIR = r"B:\mode\plan15h"
for name, script, tmax in [
    ("71_v12r_tex", r"B:\mode\vton_training\distill_v12r_tex.py", 2.6*3600),
    ("72_v12r_sample", r"B:\mode\vton_training\sample_v12r.py", 400),
    ("73_virtual_v14", r"B:\mode\pinterest\acmodus-agent-skills\3d-tryon\f6_v14_bright.py", 400),
    ("74_breed_v2i", r"B:\mode\pet\train_v2i_fast.py", 1.6*3600),
    ("75_pet_v29", r"B:\mode\pet\chain_v29.py", 900),
]:
    if not os.path.exists(script):
        open(os.path.join(LOGDIR, "_progress.txt"), "a", encoding="utf-8").write(name + ": SKIP\n")
        continue
    try:
        with open(os.path.join(LOGDIR, name + ".log"), "w", encoding="utf-8") as f:
            p = subprocess.run(["py", "-3.13", "-X", "utf8", "-u", script], stdout=f, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONUSERBASE=PYUSER), timeout=int(tmax))
        st = "OK" if p.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired:
        st = "TIMEOUT"
    open(os.path.join(LOGDIR, "_progress.txt"), "a", encoding="utf-8").write(name + ": " + st + "\n")
