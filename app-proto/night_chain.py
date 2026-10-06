
# 夜班自动收尾:SDXL教师化+宠物CN→下载→关机
import paramiko, time, os, sys, zipfile

HOST, PORT, USER, PW = "connect.bjb1.seetacloud.com", 48077, "root", "D1Zi0yJnzCza"

def sh(c, cmd, t=60):
    _, o, e = c.exec_command(cmd, timeout=t)
    return (o.read() or e.read()).decode(errors="replace")

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30)
    return c

def stage(c):
    n_teach = sh(c, "ls /root/autodl-tmp/teacher_sdxl 2>/dev/null | wc -l").strip()
    log = sh(c, "tail -c 400 /root/autodl-tmp/sdxl.log 2>/dev/null | tr '\\r' '\\n' | tail -2")
    done_sdxl = "SDXL_TEACHER_DONE" in sh(c, "cat /root/autodl-tmp/sdxl.log 2>/dev/null | grep -c SDXL_TEACHER_DONE")
    return int(n_teach or 0), done_sdxl, log

def start_pet(c):
    sh(c, "cd /root/autodl-tmp && export HF_ENDPOINT=https://hf-mirror.com PATH=/root/miniconda3/bin:$PATH && nohup python -u 10_pet_cn.py > pet.log 2>&1 & echo pet_started")

def pet_done(c):
    return "PET_CN_DONE" in sh(c, "cat /root/autodl-tmp/pet.log 2>/dev/null | grep -c PET_CN_DONE")

def pull_and_shutdown(c):
    sftp = c.open_sftp()
    os.makedirs(r"B:\\mode\\cloud_out", exist_ok=True)
    import glob as g
    # 教师图
    files = sh(c, "ls /root/autodl-tmp/teacher_sdxl/ 2>/dev/null").split()
    for f in files:
        try: sftp.get("/root/autodl-tmp/teacher_sdxl/" + f, r"B:\\mode\\cloud_out\\" + f)
        except Exception as ex: print("dl err", f, str(ex)[:40], flush=True)
    # 宠物图
    pfiles = sh(c, "ls /root/autodl-tmp/pet_out/ 2>/dev/null").split()
    for f in pfiles:
        try: sftp.get("/root/autodl-tmp/pet_out/" + f, r"B:\\mode\\cloud_out\\pet_" + f)
        except Exception as ex: print("dl err", f, str(ex)[:40], flush=True)
    sftp.close()
    n = len(g.glob(r"B:\\mode\\cloud_out\\*.png"))
    print("downloaded:", n, flush=True)
    open(r"B:\\mode\\cloud_c\\NIGHT_RESULT.txt", "w").write(f"downloaded {n} files; shutting down at " + time.strftime("%H:%M"))
    if n >= 600:  # 成品齐才关
        sh(c, "sync; shutdown -h now", t=15)
        print("SHUTDOWN_SENT", flush=True)

print("night chain start", time.strftime("%H:%M:%S"), flush=True)
pet_started = False
while True:
    try:
        c = connect()
        nt, done, log = stage(c)
        print(time.strftime("%H:%M"), "teachers:", nt, "done:", done, "|", log.replace("\n", " ")[:60], flush=True)
        if done and not pet_started:
            start_pet(c)
            pet_started = True
            print(">>> PET CN started", flush=True)
        if pet_started and pet_done(c):
            print(">>> ALL DONE, pulling + shutdown", flush=True)
            pull_and_shutdown(c)
            break
        c.close()
    except Exception as ex:
        print("ERR", str(ex)[:80], flush=True)
    time.sleep(300)
print("NIGHT_COMPLETE", flush=True)
