#!/usr/bin/env python3
"""对照组："LLM 生成整段 C 代码文本 + 外部编排"（文章里的范式）。

每轮：把算子语义、函数签名、朴素参考实现、规则、历史尝试（状态/耗时/报错摘要）和当前最优源码发给 LLM，
LLM 返回一个完整 C 文件；交给与 Aura 方法完全相同的 keval 评测。预算 = 评测次数（每个回复算一次）。
  text_baseline.py --op OP --seed S --budget B --dir RUN_DIR --out result.json
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keval, kgen, llm  # noqa: E402

SYSTEM = """你是 CPU 高性能内核工程师。为给定算子写一个 C 函数 `kernel`（签名必须完全一致），在单核 x86-64（AVX-512）上尽量快。
评测规则（违反即判作弊或失败）：
- 编译参数固定：gcc -std=gnu11 -O2 -march=native -fno-tree-vectorize -fno-unroll-loops，不得用 pragma/attribute 改优化选项
- 只能 #include：math.h string.h stddef.h stdint.h stdlib.h immintrin.h float.h limits.h stdbool.h
- 禁止：内联汇编、static 变量（static inline 函数可以）、可写全局变量、除 kernel 外的导出符号（辅助函数请写 static inline）、
  任何 I/O/计时/进程/环境变量调用；不得修改输入
- 正确性：与 double 参考比较（见语义里的精度要求），多个形状（含非 2 次幂），输入每次随机且两组交替
只输出一个 ```c 代码块。"""


def run(a):
    os.makedirs(a.dir, exist_ok=True)
    op = a.op; best = None; best_src = keval.NAIVE[op]; hist = []; t0 = time.time()
    led = open(os.path.join(a.dir, "trials.jsonl"), "w")
    for it in range(a.budget):
        user = json.dumps({"算子": op, "语义": kgen.SEMANTICS[op], "签名": kgen.SIGS[op],
                           "朴素参考实现": keval.NAIVE[op], "历史尝试": hist[-12:],
                           "当前最优源码": best_src, "当前最优毫秒": best, "剩余预算": a.budget - it}, ensure_ascii=False, indent=1)
        text, rec = llm.chat(a.provider, SYSTEM, user, os.path.join(a.dir, "llm_calls.jsonl"), tag="text-%d" % it)
        m = re.findall(r"```(?:c|C|cpp)?\s*\n(.*?)```", text, flags=re.S)
        src = m[-1] if m else ""
        if not src.strip():
            r = {"status": "no_code" if not rec.get("error") else "llm_error"}
        else:
            r = keval.evaluate(op, src)
        ms = r.get("ms_geo")
        improved = r["status"] == "ok" and (best is None or ms < best * 0.98)
        if improved:
            best, best_src = ms, src
        brief = {"轮次": it, "状态": r["status"], "毫秒": ms, "作弊标记": r.get("flags"),
                 "报错摘要": (r.get("error") or "")[-400:] or None, "max_err": r.get("max_err")}
        hist.append(brief)
        led.write(json.dumps({"it": it, "status": r["status"], "ms": ms, "flags": r.get("flags"), "improved": improved,
                              "src_sha256": r.get("src_sha256"), "eval_s": r.get("eval_s"), "llm_s": rec.get("latency_s"),
                              "usage": rec.get("usage"), "best": best}, ensure_ascii=False) + "\n"); led.flush()
        print("text %s seed=%s it=%d %s ms=%s best=%s" % (op, a.seed, it, r["status"], ms, best), flush=True)
    open(os.path.join(a.dir, "best.c"), "w").write(best_src)
    json.dump({"op": op, "seed": a.seed, "best_ms": best, "wall_s": round(time.time() - t0, 1)}, open(a.out, "w"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True); p.add_argument("--seed", type=int, default=1); p.add_argument("--budget", type=int, default=16)
    p.add_argument("--dir", required=True); p.add_argument("--out", required=True)
    p.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "deepseek"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
