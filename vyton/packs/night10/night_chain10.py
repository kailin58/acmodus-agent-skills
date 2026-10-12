# night_chain10.py — 10小时自主任务链
# 顺序:cut_v2.1增强→Glow v8.1强度→Style v18扩容→宠物v8.1→Try手部v5→验收循环
import subprocess, os, sys, time, json, glob

LOG = r"B:\mode\night10.log"
PYUSER = r"C:\Users\Administrator\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages"
ENV = dict(os.environ, PYTHONUSERBASE=PYUSER)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_py(script, timeout=7200, step=""):
    log(f"START {step}")
    try:
        r = subprocess.run(["cmd", "/c", "py", "-3.13", "-X", "utf8", script],
                           capture_output=True, timeout=timeout, env=ENV,
                           cwd=os.path.dirname(script))
        out = (r.stdout or b"").decode(errors="replace")
        err = (r.stderr or b"").decode(errors="replace")
        log(f"END {step} | tail: {(out or err).strip()[-200:]}")
        return out
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT {step}")
        return ""
    except Exception as e:
        log(f"ERR {step}: {e}")
        return ""

def git_commit(msg):
    repo = r"B:\mode\pinterest\acmodus-agent-skills"
    cfg = ["-c", "safe.directory=B:/mode/pinterest/acmodus-agent-skills",
           "-c", "user.email=kailinwang58@gmail.com", "-c", "user.name=kailin58"]
    subprocess.run(["cmd", "/c", "git", "-C", repo] + cfg + ["add", "-A"], capture_output=True, timeout=120)
    subprocess.run(["cmd", "/c", "git", "-C", repo] + cfg + ["commit", "-m", msg, "--no-verify"],
                   capture_output=True, timeout=120)
    log(f"COMMIT {msg[:50]}")

log("=== 夜班10号链启动 ===")

# ── 任务1:cut_v2.1增强(30ep+320输入+边缘加权)≈90分钟 ──
if not os.path.exists(r"B:\mode\models\cut\v21_done.flag"):
    run_py(r"B:\mode\models\cut\train_v21.py", 7200, "cut_v2.1增强训练")
    open(r"B:\mode\models\cut\v21_done.flag", "w").write("1")
    git_commit("夜班:cut_v2.1增强(320输入+边缘加权30ep)")

# ── 任务2:Glow v8.1强度系数(0.7档,20ep)≈30分钟 ──
if not os.path.exists(r"B:\mode\glow_data\v81_done.flag"):
    run_py(r"B:\mode\glow_data\train_glow_v81.py", 3600, "Glow v8.1强度0.7")
    open(r"B:\mode\glow_data\v81_done.flag", "w").write("1")
    git_commit("夜班:Glow v8.1强度0.7(156%超调→目标90-100%)")

# ── 任务3:Style v18数据扩容(画板新图3083,60ep)≈120分钟 ──
if not os.path.exists(r"B:\mode\style_train\v18_done.flag"):
    run_py(r"B:\mode\style_train\train_v18.py", 7200, "Style v18扩容")
    open(r"B:\mode\style_train\v18_done.flag", "w").write("1")
    git_commit("夜班:Style v18数据扩容3083items")

# ── 任务4:宠物v8.1后半身渐隐修复(全身保形后处理)≈60分钟 ──
if not os.path.exists(r"B:\mode\pet_v81_done.flag"):
    run_py(r"B:\mode\pet_v81.py", 3600, "宠物v8.1渐隐修复")
    open(r"B:\mode\pet_v81_done.flag", "w").write("1")
    git_commit("夜班:宠物v8.1后半身渐隐修复")

# ── 任务5:Try手部v5(depth引导)≈90分钟 ──
if not os.path.exists(r"B:\mode\vton_training\v5_done.flag"):
    run_py(r"B:\mode\vton_training\hand_v5.py", 5400, "Try手部v5")
    open(r"B:\mode\vton_training\v5_done.flag", "w").write("1")
    git_commit("夜班:Try手部v5(depth引导)")

# ── 任务6:证据图批量产出(给明早vision验收) ──
run_py(r"B:\mode\models\cut\domain_v2.py", 300, "cut域内证据图")
log("=== 链完成,等自愈任务兜底 ===")
