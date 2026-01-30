# -*- coding: utf-8 -*-
"""
login_hammer_200qps.py   (2025-07-13)

以单账号固定密码 200 QPS 持续压测登录接口，并实时输出仪表盘。
"""

import asyncio, aiohttp, json, base64, urllib.parse, time, logging, datetime
from aiolimiter import AsyncLimiter
from tqdm import tqdm
from statistics import mean

# ---------- 参数 ----------
TARGET_URL          = "https://view.shufe.edu.cn/api/v1/user/sign-in"
TEST_USERNAME       = "2015000031"          # 固定账号
TEST_PASSWORD       = "123456"              # 固定密码
MAX_QPS             = 200                   # **改为 200**
MAX_CONCURRENT_TASK = 20                    # 单协程请求数（随意，只要够用）
RETRY_LIMIT, RETRY_BACKOFF = 3, 0.5         # 与原脚本一致

# ---------- 日志 ----------
logging.basicConfig(
    filename='login_hammer.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# ---------- 限流 & 统计 ----------
limiter          = AsyncLimiter(MAX_QPS, time_period=1)
stats_lock       = asyncio.Lock()
total_requests   = total_errors = total_retry_fail = 0
sec_requests     = sec_errors = 0
sec_latencies    = []
script_start     = datetime.datetime.now()
status_bar       = tqdm(total=0, bar_format='{desc}', leave=True, position=0)

# ---------- 工具函数 ----------
def mixToken(token, force=True, encrypt_token=True, fixed_timestamp=None):
    if encrypt_token or force:
        ts   = str(fixed_timestamp) if fixed_timestamp else str(int(time.time()*1000))
        enc  = urllib.parse.quote(token) + '_' + ts
        rev  = enc[::-1]
        fin  = base64.b64encode(rev.encode()).decode()
        return (base64.b64encode((fin + '_' + ts).encode()).decode())[::-1]
    return token

def fmt_timespan(delta: datetime.timedelta) -> str:
    s = int(delta.total_seconds())
    d, s = divmod(s, 86400);  h, s = divmod(s, 3600);  m, s = divmod(s, 60)
    return f"{d}d {h:02d}:{m:02d}:{s:02d}"

async def send_sign_in_request(session):
    global total_requests, total_errors, total_retry_fail
    global sec_requests, sec_errors, sec_latencies

    payload = {"c": mixToken(f"{TEST_USERNAME}:{TEST_PASSWORD}", force=True)}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    for attempt in range(1, RETRY_LIMIT + 1):
        start_tp = time.perf_counter()
        try:
            async with limiter:
                async with session.post(TARGET_URL, json=payload, headers=headers, timeout=10) as r:
                    status = r.status
                    _ = await r.text()            # 不关心业务结果，只需状态码
            latency = (time.perf_counter() - start_tp) * 1000

            async with stats_lock:
                total_requests += 1
                sec_requests   += 1
                sec_latencies.append(latency)
                if status != 200:
                    total_errors += 1
                    sec_errors   += 1
            return
        except Exception:
            latency = (time.perf_counter() - start_tp) * 1000
            async with stats_lock:
                total_requests += 1
                total_errors   += 1
                sec_requests   += 1
                sec_errors     += 1
                sec_latencies.append(latency)
                if attempt == RETRY_LIMIT:
                    total_retry_fail += 1
            await asyncio.sleep(RETRY_BACKOFF * attempt)

async def worker(session):
    """不停地提交请求，直到程序被手动终止"""
    while True:
        await send_sign_in_request(session)

async def live_status(refresh_interval=1):
    global sec_requests, sec_errors, sec_latencies
    while True:
        await asyncio.sleep(refresh_interval)
        async with stats_lock:
            qps    = sec_requests / refresh_interval
            avg_rt = mean(sec_latencies) if sec_latencies else 0

            sec_requests = sec_errors = 0
            sec_latencies.clear()

            elapsed = datetime.datetime.now() - script_start
            status_bar.set_description_str(
                f"RUN {fmt_timespan(elapsed)} | "
                f"QPS {qps:6.1f} | "
                f"REQ {total_requests:,} | "
                f"ERR {total_errors} (retry-fail {total_retry_fail}) | "
                f"AVG {avg_rt:4.0f} ms"
            )
            status_bar.refresh()

async def main():
    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=None, keepalive_timeout=15)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # 启动实时仪表盘
        status_task = asyncio.create_task(live_status())

        # 启动若干 worker 并发打点
        workers = [asyncio.create_task(worker(session)) for _ in range(MAX_CONCURRENT_TASK)]
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            pass
        finally:
            status_task.cancel()
            try:
                await status_task
            except asyncio.CancelledError:
                pass
            status_bar.close()
            logging.info("压测结束")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        status_bar.close()
        print("\n手动中断 – 已停止压测")
