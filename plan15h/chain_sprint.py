import subprocess, os
PYUSER = r"C:\Users\Administrator\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages"
LOGDIR = r"B:\mode\plan15h"
for name, script, tmax in [
    ("61_v11r_edge", r"B:\mode\vton_training\distill_v11r_edge.py", 2.6*3600),
    ("62_v11r_sample", r"B:\mode\vton_training\sample_v11r.py", 400),
    ("63_v2h_s1", r"B:\mode\pet\train_v2h_s1.py", 2.0*3600),
    ("64_v2h_s2", r"B:\mode\pet\train_v2h_s2.py", 1.2*3600),
    ("65_virtual_v13", r"B:\mode\pinterest\acmodus-agent-skills\3d-tryon\f6_v13_auburn70.py", 400),
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
