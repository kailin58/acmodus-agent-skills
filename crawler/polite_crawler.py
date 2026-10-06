# -*- coding: utf-8 -*-
"""ACmodus礼貌采集器:频率低于目标站反爬阈值(用户铁律)
用法: from polite_crawler import PoliteCrawler; c=PoliteCrawler("example.com")
"""
import json, os, random, time, subprocess

LIMITS_FILE = r"B:\mode\crawl_limits.json"

class PoliteCrawler:
    def __init__(self, domain):
        self.domain = domain
        self.limits = json.load(open(LIMITS_FILE, encoding="utf-8")) if os.path.exists(LIMITS_FILE) else {}
        d = self.limits.get(domain, {})
        self.min_gap = d.get("min_gap", random.uniform(5, 15))   # 秒,随机
        self.last_req = 0.0
        self.session_count = 0
        self.max_session = d.get("max_session", 50)              # 单次上限

    def _gap(self):
        wait = self.min_gap + random.uniform(0, 3)  # 抖动
        dt = time.time() - self.last_req
        if dt < wait: time.sleep(wait - dt)
        self.last_req = time.time()

    def fetch(self, url, out=None, proxy=True):
        """curl抓取/下载,自带节流+计数+429检测"""
        if self.session_count >= self.max_session:
            raise RuntimeError(f"达到单站单次上限{self.max_session},今日到此为止(铁律)")
        self._gap()
        self.session_count += 1
        args = ["curl", "-s", "-m", "25",
                "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"]
        if proxy: args += ["-x", "socks5h://127.0.0.1:7897"]
        if out: args += ["-o", out, "-w", "%{http_code}"]
        else: args += ["-w", "\n%{http_code}"]
        args.append(url)
        r = subprocess.run(args, capture_output=True, text=True, timeout=40)
        code = (r.stdout or "").strip().splitlines()[-1] if r.stdout else "000"
        if code in ("429", "403"):
            self._record_block(code)
            raise RuntimeError(f"被限流{code},立即冷却2小时(已记录)")
        return r.stdout if not out else code

    def _record_block(self, code):
        self.limits.setdefault(self.domain, {})
        self.limits[self.domain].update({"blocked": code, "blocked_at": time.strftime("%F %T"),
                                          "cooldown_until": time.strftime("%F %T", time.localtime(time.time()+7200))})
        json.dump(self.limits, open(LIMITS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
