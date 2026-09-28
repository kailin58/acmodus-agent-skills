import subprocess, os
PYUSER = r"C:\Users\Administrator\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages"
LOGDIR = r"B:\mode\plan15h"
for name, script, tmax in [("13_try_v6f_3000ep", r"B:\mode\vton_training\distill_v6f.py", 4.2*3600),
                            ("14_try_v6f_sample", r"B:\mode\vton_training\sample_v6f.py", 300)]:
    log = os.path.join(LOGDIR, name + ".log")
    try:
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.run(["py", "-3.13", "-X", "utf8", script], stdout=f, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONUSERBASE=PYUSER), timeout=int(tmax))
        st = "OK" if p.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired:
        st = "TIMEOUT"
    open(os.path.join(LOGDIR, "_progress.txt"), "a", encoding="utf-8").write(name + ": " + st + "\n")
