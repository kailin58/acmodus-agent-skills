import subprocess, os, json
PYUSER = r"C:\Users\Administrator\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages"
LOGDIR = r"B:\mode\plan15h"
TASKS = [
    # v9r/v2g已在跑,链只管后续
    ("51_v10r_multihead", r"B:\mode\vton_training\distill_v10r_multi.py", 2.5*3600),
    ("52_v10r_sample", r"B:\mode\vton_training\sample_v10r.py", 400),
    ("53_style_v4i", r"B:\mode\style_train\train_v4i.py", 1.2*3600),
    ("54_gender_v7_ens", r"B:\mode\face_data\train_gender_v7.py", 1.3*3600),
    ("55_glow_v4", r"B:\mode\train_glow4.py", 1800),
    ("56_pet_v28", r"B:\mode\pet\chain_v28.py", 900),
    ("57_rsi_rules", r"B:\mode\rsi\auto_tuner.py rules", 60),
]
for name, script, tmax in TASKS:
    if " " in script:  # 带参数命令
        parts = script.split()
        try:
            with open(os.path.join(LOGDIR, name + ".log"), "w", encoding="utf-8") as f:
                p = subprocess.run(["py", "-3.13", "-X", "utf8"] + parts, stdout=f, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONUSERBASE=PYUSER), timeout=int(tmax))
            st = "OK" if p.returncode == 0 else "FAIL"
        except subprocess.TimeoutExpired:
            st = "TIMEOUT"
    else:
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
# 终局账本更新
print("FINAL_CHAIN_DONE")
