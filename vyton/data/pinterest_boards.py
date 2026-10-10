# 用户画板采集:yonglinwang3804全48板
import subprocess, re, json, os, time, urllib.parse

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
USER = "yonglinwang3804"
IDX = r"B:\mode\pinterest\board_hashes.json"

# 先抓全部画板列表
r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                    "--user-data-dir=B:\\mode\\pinterest\\chrome_profile",
                    "--proxy-server=socks5://127.0.0.1:7897",
                    "--window-size=1400,4000",
                    "--dump-dom", "--virtual-time-budget=25000", "--timeout=45000",
                    f"https://www.pinterest.com/{USER}/_boards/"],
                   capture_output=True, timeout=90)
dom = r.stdout.decode("utf-8", errors="replace")
boards = sorted(set(re.findall(rf'/{USER}/([a-z0-9%-]{{3,40}})/', dom)))
boards = [b for b in boards if not b.startswith("_")]
    HARD_LIST = ['%E6%AC%BE%E5%BC%8F%E6%94%B6%E8%97%8F', '%e5%86%85%e6%90%ad', '%e5%86%85%e8%a1%a3', '%e5%88%9b%e6%96%b0', '%e5%8e%9f%e5%9e%8b', '%e5%8e%bb%e6%b1%a1', '%e5%8f%a3%e7%bd%a9%e8%ae%be%e8%ae%a1', '%e5%a4%a7%e8%a1%a3', '%e5%ad%95%e5%a6%87%e8%a3%85', '%e5%ae%b6%e5%b1%85%e8%a3%85%e6%bd%a2', '%e5%b8%bd%e5%ad%90%e5%86%85%e8%a3%a4', '%e5%bb%ba%e7%ad%91', '%e6%89%8b%e5%a5%97', '%e6%89%8b%e5%b7%a5', '%e6%91%84%e5%bd%b1%e6%8a%80%e5%b7%a7', '%e6%9c%8d%e8%a3%85', '%e6%9c%8d%e8%a3%85%e5%90%8a%e7%89%8c', '%e6%9c%8d%e8%a3%85%e5%93%81%e7%89%8c', '%e6%9c%8d%e8%a3%85%e5%b8%bd%e5%ad%90', '%e6%9c%8d%e8%a3%85%e6%8a%80%e6%9c%af', '%e6%9c%8d%e8%a3%85%e7%bb%86%e8%8a%82']
    boards = sorted(set(boards + HARD_LIST))
    HARD = []
    boards = sorted(set(boards + HARD))
    if len(boards) < 40:
        boards = sorted(set(list(out.keys()) + boards))  # 用已存的+新抓的合并
print("boards:", len(boards), flush=True)

out = {}
if os.path.exists(IDX):
    out = json.load(open(IDX, encoding="utf-8"))
for b in boards:
    if b in out:
        continue
    r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--user-data-dir=B:\\mode\\pinterest\\chrome_profile",
                        "--proxy-server=socks5://127.0.0.1:7897",
                        "--window-size=1400,5000",
                        "--dump-dom", "--virtual-time-budget=30000", "--timeout=50000",
                        f"https://www.pinterest.com/{USER}/{b}/"],
                       capture_output=True, timeout=100)
    d = r.stdout.decode("utf-8", errors="replace")
    hs = sorted(set(re.findall(r'i\.pinimg\.com/\d+x(?:/[0-9a-f]{2}){3}/([0-9a-f]{24,40})\.', d)))
    out[b] = hs
    json.dump(out, open(IDX, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{urllib.parse.unquote(b)}: {len(hs)}", flush=True)
    time.sleep(6)
tot = sum(len(v) for v in out.values())
print("BOARDS_DONE", tot, flush=True)
