# -*- coding: utf-8 -*-
# 15小时自动训练链:顺序执行,独立日志,失败跳过
import subprocess, os, time

PYUSER = r"C:\Users\Administrator\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages"
LOGDIR = r"B:\mode\plan15h"

TASKS = [
    ("01_try_v6d_1500ep", r"B:\mode\vton_training\distill_v6d.py", 3.0*3600),
    ("02_try_v6d_sample", r"B:\mode\vton_training\sample_v6d.py", 300),
    ("03_breed_v2d_enb3", r"B:\mode\pet\train_breed_v2d.py", 1.8*3600),
    ("04_pet_lora_v4_data", r"B:\mode\pet\gen_lora_v4.py", 1800),
    ("05_pet_lora_v4_train", r"B:\mode\pet\train_pet_lora_v4.py", 1.2*3600),
    ("06_gender_v5", r"B:\mode\face_data\train_gender_v5.py", 1.2*3600),
    ("07_style_v4g_color", r"B:\mode\style_train\train_v4g.py", 3600),
    ("08_glow_v3", r"B:\mode\train_glow3.py", 1800),
    ("09_try_v6e_1500ep", r"B:\mode\vton_training\distill_v6e.py", 3.0*3600),
    ("10_try_v6e_sample", r"B:\mode\vton_training\sample_v6e.py", 300),
]

def run_all():
    t_start = time.time()
    summary = []
    for name, script, tmax in TASKS:
        if not os.path.exists(script):
            summary.append(name + ": SKIP(无脚本)")
            continue
        log = os.path.join(LOGDIR, name + ".log")
        env = dict(os.environ, PYTHONUSERBASE=PYUSER)
        t0 = time.time()
        try:
            with open(log, "w", encoding="utf-8") as f:
                p = subprocess.run(["py", "-3.13", "-X", "utf8", script],
                                   stdout=f, stderr=subprocess.STDOUT,
                                   env=env, timeout=int(tmax))
            ok = (p.returncode == 0)
        except subprocess.TimeoutExpired:
            ok = None
        except Exception as e:
            ok = False
            with open(log, "a", encoding="utf-8") as f:
                f.write("\n[chain-err]" + str(e))
        dt = (time.time()-t0)/60
        st = "OK" if ok else ("TIMEOUT" if ok is None else "FAIL")
        summary.append(name + ": " + st + " " + str(int(dt)) + "min")
        with open(os.path.join(LOGDIR, "_progress.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(summary) + "\n[elapsed " + str(round((time.time()-t_start)/3600,1)) + "h]")
    with open(os.path.join(LOGDIR, "_final.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

run_all()
