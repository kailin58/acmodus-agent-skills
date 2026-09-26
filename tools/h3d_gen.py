# 云端3D资产生成工具(代号H3D,OAuth已授权)
# 用法:python h3d_gen.py <图片路径> [输出.glb]
import subprocess, sys, os, json, time
NODE = r"B:\HERMES AGENT\node\node.exe"
CLI = r"C:\WINDOWS\system32\config\systemprofile\AppData\Roaming\npm\node_modules\@hyper3d\cli\dist\index.js"
ENV = dict(os.environ, HTTPS_PROXY="http://127.0.0.1:7897", HTTP_PROXY="http://127.0.0.1:7897")

def gen(img, tier="Gen-2.5-Medium"):
    r = subprocess.run([NODE, CLI, "generate", "--image", img, "--tier", tier,
                        "--format", "glb", "--output", "json"], capture_output=True, env=ENV, timeout=120)
    gid = json.loads(r.stdout)["generation_id"]
    for _ in range(60):
        time.sleep(10)
        s = json.loads(subprocess.run([NODE, CLI, "status", gid, "--output", "json"],
                       capture_output=True, env=ENV, timeout=60).stdout)
        if s["status"] == "completed": break
    r2 = subprocess.run([NODE, CLI, "result", gid, "--output", "json"], capture_output=True, env=ENV, timeout=60)
    url = json.loads(r2.stdout)["files"][0]["url"]
    dst = sys.argv[2] if len(sys.argv) > 2 else img.rsplit(".",1)[0] + "_h3d.glb"
    subprocess.run(["curl", "-sL", "-m", "240", "-x", "http://127.0.0.1:7897", "-o", dst, url], timeout=260)
    return dst

if __name__ == "__main__":
    print("GLB:", gen(sys.argv[1]))
