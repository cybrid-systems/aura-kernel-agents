#!/usr/bin/env python3
"""汇总：收敛曲线、无效试错、最终复测、相对编译器基线与 SOL 类比分数。

  summarize.py remeasure --runs runs --out results/final.tsv   # 每个运行的最终内核用新随机输入复测 3 次取中位（在 dev 镜像里跑）
  summarize.py report --runs runs --final results/final.tsv --calib results/calib.json --base results/compiler_baseline.json --out results/summary.md
"""
import argparse, csv, glob, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INVALID = {"compile_error", "incorrect", "hack", "crash", "timeout", "no_code", "llm_error", "lower-error", "aura-error"}


def load_trials(d):
    """-> 按评测顺序的 [(status, ms)]（不含起点与确认复测），以及计数"""
    out = []; pre = {"illegal": 0, "dup": 0}
    if os.path.exists(os.path.join(d, "trials.tsv")):
        for r in csv.DictReader(open(os.path.join(d, "trials.tsv")), delimiter="\t"):
            if r["mutation"] in ("start", "confirm"):
                continue
            if r["status"] in ("illegal", "dup"):
                pre[r["status"]] += 1; continue
            out.append((r["status"], float(r["ms"]) if r["ms"] else None))
    else:
        for line in open(os.path.join(d, "trials.jsonl")):
            r = json.loads(line); out.append((r["status"], r["ms"]))
    return out, pre


def runs(root):
    for d in sorted(glob.glob(os.path.join(root, "*", "*", "s*"))):
        op, m, s = d.split(os.sep)[-3:]
        if os.path.exists(os.path.join(d, "trials.tsv")) or os.path.exists(os.path.join(d, "trials.jsonl")):
            yield op, m, int(s[1:]), d


def final_src(d, op):
    import kgen
    if os.path.exists(os.path.join(d, "best.ast")):
        return kgen.lower(open(os.path.join(d, "best.ast")).read().strip())[1]
    return open(os.path.join(d, "best.c")).read()


def cmd_remeasure(a):
    import keval
    w = open(a.out, "w"); w.write("op\tmethod\tseed\tfinal_ms\tsamples\n")
    for op, m, s, d in runs(a.runs):
        if not (os.path.exists(os.path.join(d, "best.ast")) or os.path.exists(os.path.join(d, "best.c"))):
            continue
        src = final_src(d, op); xs = []
        for _ in range(3):
            r = keval.evaluate(op, src)
            if r["status"] == "ok":
                xs.append(r["ms_geo"])
        w.write("%s\t%s\t%d\t%s\t%s\n" % (op, m, s, statistics.median(xs) if xs else "", ",".join("%.5f" % x for x in xs))); w.flush()
        print(op, m, s, xs, flush=True)


def sol_score(t, tb, ts):
    return (tb - ts) / ((t - ts) + (tb - ts)) if tb > ts else float("nan")


def cmd_report(a):
    fin = {(r["op"], r["method"], int(r["seed"])): float(r["final_ms"]) for r in csv.DictReader(open(a.final), delimiter="\t") if r["final_ms"]}
    calib = json.load(open(a.calib)); base = json.load(open(a.base))
    data = {}
    for op, m, s, d in runs(a.runs):
        tr, pre = load_trials(d); data.setdefault(op, {}).setdefault(m, []).append((s, tr, pre, d))
    L = []
    for op in data:
        best_any = min(ms for m in data[op] for (_, tr, _, _) in data[op][m] for st, ms in tr if st == "ok" and ms)
        tb = base[op]["ms_geo"]; ts = calib["ops"][op]["t_sol_ms_geo"]
        start = next(iter(json.load(open(f)) for f in glob.glob(os.path.join(a.runs, op, "aura-*", "s1", "start-0", "res.json"))), {}).get("ms_geo")
        L.append("\n### %s\n" % op)
        L.append("起点（朴素调度，-O2 无自动向量化）%.4f ms；编译器基线（朴素源码 -O3 -ffast-math 自动向量化，T_b）%.4f ms；单核 roofline T_SOL %.4f ms。\n" % (start or float("nan"), tb, ts))
        L.append("| 方法 | 种子 | 最终复测 ms（中位） | ×起点 | ×编译器基线 | SOL 类比分 | 4/8/12/16 次评测时的最优 ms | 达到全方法最优 1.1× 内所需评测 | 无效评测（编译错/错误/作弊/其他） | 契约拦截+去重（不耗预算） |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for m in sorted(data[op]):
            rows = data[op][m]; fs = [fin.get((op, m, s)) for s, *_ in rows]; fs = [f for f in fs if f]
            curve = []; reach = []; inv = {"compile_error": 0, "incorrect": 0, "hack": 0, "other": 0}; pre_t = 0
            for s, tr, pre, d in rows:
                b = None; cs = []; rc = None
                for i, (st, ms) in enumerate(tr):
                    if st == "ok" and ms and (b is None or ms < b):
                        b = ms
                    if rc is None and b is not None and b <= 1.1 * best_any:
                        rc = i + 1
                    if st in INVALID:
                        inv[st if st in inv else "other"] += 1
                    cs.append(b)
                curve.append(cs); reach.append(rc); pre_t += pre["illegal"] + pre["dup"]
            def at(k):
                v = [c[k - 1] for c in curve if len(c) >= k and c[k - 1]]
                return "%.3f" % statistics.mean(v) if v else "–"
            fm = statistics.mean(fs) if fs else float("nan")
            L.append("| %s | %d | %.4f | %.1f | %.2f | %.2f | %s | %s | %d/%d/%d/%d | %d |" % (
                m, len(rows), fm, (start or fm) / fm, tb / fm, sol_score(fm, tb, ts),
                " / ".join(at(k) for k in (4, 8, 12, 16)), ",".join(str(r) if r else "未达" for r in reach),
                inv["compile_error"], inv["incorrect"], inv["hack"], inv["other"], pre_t))
    L.append("\n### 成本（每次运行平均）\n")
    L.append("| 算子 | 方法 | 墙钟 s | LLM 调用 | LLM token（总） | LLM 平均延迟 s |")
    L.append("|---|---|---|---|---|---|")
    aw = {}
    if os.path.exists(a.aura):
        for r in csv.DictReader(open(a.aura), delimiter="\t"):
            aw.setdefault((r["op"], r["method"]), []).append(float(r["wall_ms"]) / 1000)
    for op in data:
        for m in sorted(data[op]):
            ws, calls, toks, lat = [], 0, 0, []
            for s, tr, pre, d in data[op][m]:
                rj = os.path.join(d, "result.json")
                if os.path.exists(rj):
                    ws.append(json.load(open(rj))["wall_s"])
                lj = os.path.join(d, "llm_calls.jsonl")
                if os.path.exists(lj):
                    for line in open(lj):
                        r = json.loads(line); calls += 1; toks += (r.get("usage") or {}).get("total_tokens", 0); lat.append(r["latency_s"])
            ws = ws or aw.get((op, m), []); n = len(data[op][m])
            L.append("| %s | %s | %.1f | %.1f | %.0f | %s |" % (op, m, statistics.mean(ws) if ws else float("nan"), calls / n, toks / n,
                                                         "%.1f" % statistics.mean(lat) if lat else "–"))
    open(a.out, "w").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("remeasure"); s.add_argument("--runs", default="runs"); s.add_argument("--out", required=True)
    s = sp.add_parser("report")
    for k in ("runs", "final", "calib", "base", "out"):
        s.add_argument("--" + k, required=True)
    s.add_argument("--aura", default="results/aura_runs.tsv")
    a = p.parse_args(); {"remeasure": cmd_remeasure, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
