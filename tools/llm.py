"""OpenAI 兼容的 LLM 调用（key 只从环境变量读，永不写日志/输出）。真实调用需显式 LLM_ALLOW_NETWORK=1。"""
import hashlib, json, os, re, time, urllib.request

PROVIDERS = {
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-flash"),
    "minimax": ("MINIMAX_API_KEY", "https://api.minimaxi.com/v1", "MiniMax-M3.1-Flash"),
    "glm": ("GLM_API_KEY", "https://open.bigmodel.cn/api/coding/paas/v4", "glm-5.3-flash"),
}


def chat(provider, system, user, log_path, tag="", max_tokens=16000, timeout=300):
    key_env, base, model = PROVIDERS[provider]
    rec = {"t": time.time(), "provider": provider, "model": model, "tag": tag,
           "prompt_sha256": hashlib.sha256((system + user).encode()).hexdigest(), "prompt_chars": len(system) + len(user)}
    t0 = time.time(); text = ""
    try:
        key = os.environ.get(key_env, "")
        if not key:
            raise RuntimeError("缺少环境变量 %s" % key_env)
        if os.environ.get("LLM_ALLOW_NETWORK") != "1":
            raise RuntimeError("未设置 LLM_ALLOW_NETWORK=1")
        body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens}
        if provider == "deepseek":   # 推理模型会把 16k 预算全花在 reasoning 上（实测 no_code），默认关闭思考（off）；也可设 low/medium，但实测 low 对难题仍会烧满预算
            eff = os.environ.get("LLM_REASONING", "off"); rec["reasoning"] = eff
            if eff == "off":
                body["thinking"] = {"type": "disabled"}
            else:
                body["reasoning_effort"] = eff
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        last = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    resp = json.loads(r.read().decode()); break
            except Exception as e:   # 报错里不带请求头
                last = "%s %s" % (type(e).__name__, getattr(e, "code", "")); time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError("请求失败：" + last)
        text = resp["choices"][0]["message"].get("content") or ""
        rec["usage"] = resp.get("usage")
    except Exception as e:
        rec["error"] = str(e)[:300]
    rec["latency_s"] = round(time.time() - t0, 2); rec["raw"] = text[:30000]
    with open(log_path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S), rec
