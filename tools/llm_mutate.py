#!/usr/bin/env python3
"""aura-llm 方法里 LLM 的角色：只提议"改哪些调度旋钮"，不写代码。合法性检查与应用都在 Aura 侧（契约）。

  llm_mutate.py --op OP --cur "(ast...)" --hist hist.tsv --width W --left K --dir RUN_DIR --out props.json
输出 {"proposals":[{"knob":value,...},...], "fallback":bool}；任何错误 -> 空列表（Aura 回退随机变异并记 fallback）。
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kgen, llm  # noqa: E402

SYSTEM = """你在为一个 CPU 内核（单核，x86-64 AVX-512，gcc -O2 -fno-tree-vectorize，向量化只能来自调度里的 vec 旋钮）选调度。
算子语义固定，你只能改"调度旋钮"。每个候选是对当前 incumbent 的若干旋钮修改。目标：在正确性门通过的前提下，几何平均运行时间最小。
注意：数值精度门很严格（例如 layernorm 有均值约 300 的行，reduce 要求相对误差 ≤1e-6）。不要提议已经评估过的组合。
只输出 JSON：{"proposals":[{"changes":{"<旋钮>":<取值>,...},"reason":"<一句话>"}]}。order 用字符串，如 "ikj"。"""


def main():
    p = argparse.ArgumentParser()
    for k in ("op", "cur", "hist", "dir", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--width", type=int, default=2); p.add_argument("--left", type=int, default=0)
    a = p.parse_args()
    out = {"proposals": [], "fallback": True}
    try:
        op = a.op
        hist = open(a.hist).read() if os.path.exists(a.hist) else ""
        user = json.dumps({"算子": op, "语义": kgen.SEMANTICS[op], "旋钮与合法取值": kgen.SPACE[op],
                           "当前 incumbent": a.cur, "已评估（ast / 状态 / 毫秒）": hist.strip().splitlines()[-40:],
                           "本轮提议个数": a.width, "剩余评估预算": a.left}, ensure_ascii=False, indent=1)
        text, rec = llm.chat(os.environ.get("LLM_PROVIDER", "deepseek"), SYSTEM, user, os.path.join(a.dir, "llm_calls.jsonl"), tag="mutate")
        m = re.search(r"\{.*\}", text, flags=re.S)
        props = []
        for pr in json.loads(m.group(0))["proposals"][: a.width]:
            ch = pr.get("changes", pr)
            props.append({str(k): (v if isinstance(v, int) else str(v)) for k, v in ch.items()})
        out = {"proposals": props, "fallback": not props}
    except Exception as e:
        out["error"] = type(e).__name__
    tmp = a.out + ".tmp"; json.dump(out, open(tmp, "w"), ensure_ascii=False); os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
