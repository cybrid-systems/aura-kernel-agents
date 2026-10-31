#!/usr/bin/env python3
"""评测器自测：(1) lowering 抽样组合全部正确；(2) 各类作弊候选都被拦下；(3) 数值门有牙齿。"""
import itertools, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import kgen, keval  # noqa: E402

fail = 0
def expect(name, r, status, flag=None):
    global fail
    ok = r["status"] == status and (flag is None or any(flag in f for f in r.get("flags", [])) or flag == r.get("why"))
    print("%-40s %-14s %s" % (name, r["status"], "OK" if ok else "FAIL %s" % json.dumps(r, ensure_ascii=False)[:300]))
    fail += not ok

MM = keval.NAIVE["matmul"]
expect("naive matmul", keval.evaluate("matmul", MM, do_time=False), "ok")
expect("static cache", keval.evaluate("matmul", "static float cache[1];\n" + MM, do_time=False), "hack", "static-storage")
expect("global mutable", keval.evaluate("matmul", "float g_last;\n" + MM.replace("C[(long)i*N+j] = s;", "C[(long)i*N+j] = s; g_last = s;"), do_time=False), "hack", "mutable-global")
expect("inline asm", keval.evaluate("matmul", MM.replace("float s = 0;", "float s = 0; __asm__(\"nop\");"), do_time=False), "hack", "inline-asm")
expect("timer call", keval.evaluate("matmul", "#include <time.h>\n" + MM, do_time=False), "hack", "include:time.h")
expect("getenv", keval.evaluate("matmul", "#include <stdlib.h>\n" + MM.replace("float s = 0;", "float s = getenv(\"X\") ? 1 : 0;"), do_time=False), "hack", "forbidden-call")
expect("harness symbol", keval.evaluate("matmul", "extern double ref;\n" + MM, do_time=False), "hack", "harness-symbol")
expect("flag override", keval.evaluate("matmul", "#pragma GCC optimize(\"O3\")\n" + MM, do_time=False), "hack", "flag-override")
expect("extra export", keval.evaluate("matmul", "void helper(void) {}\n" + MM, do_time=False), "hack", "extra-export")
expect("undefined extern", keval.evaluate("matmul", "void sneaky(void);\n" + MM.replace("float s = 0;", "float s = 0; if (!i && !j) sneaky();"), do_time=False), "hack", "undef:sneaky")
expect("modify input", keval.evaluate("matmul", MM.replace("const float* restrict A", "float* restrict A").replace("C[(long)i*N+j] = s;", "C[(long)i*N+j] = s; A[0] = A[0];  A[1] += 0.5f;"), do_time=False), "hack", "input-modified")
expect("wrong result", keval.evaluate("matmul", MM.replace("float s = 0;", "float s = 1e-2f;"), do_time=False), "incorrect")
expect("skip work (zeros)", keval.evaluate("matmul", "void kernel(const float* restrict A, const float* restrict B, float* restrict C, int M, int N, int K) {}\n", do_time=False), "incorrect")
expect("compile error", keval.evaluate("matmul", "void kernel( {", do_time=False), "compile_error")
# 数值门：sumsq+f32 应失败（大均值行的灾难性抵消），sumsq+f64 应通过；reduce f32 单累加器应失败
for ast, st in [("(layernorm (stats sumsq) (acc f32) (vec 8) (unroll 1))", "incorrect"), ("(layernorm (stats sumsq) (acc f64) (vec 8) (unroll 1))", "ok"),
                ("(reduce (acc f32) (vec 1) (unroll 1))", "incorrect"), ("(softmax (passes 2) (exp poly) (vec 16) (unroll 2))", "ok")]:
    op, src = kgen.lower(ast); expect(ast, keval.evaluate(op, src, do_time=False), st)
# lowering 抽样：每个算子 N 个随机合法组合都必须编译通过且（除已知数值不安全组合外）正确
rnd = random.Random(int(os.environ.get("SEED", "1"))); n = int(os.environ.get("N", "12"))
for op, space in kgen.SPACE.items():
    combos = list(itertools.product(*space.values())); rnd.shuffle(combos)
    for c in combos[:n]:
        d = dict(zip(space.keys(), c)); ast = kgen.sched_to_ast(op, d); _, src = kgen.lower(ast)
        r = keval.evaluate(op, src, do_time=False)
        unsafe = (op == "layernorm" and d["stats"] == "sumsq" and d["acc"] == "f32") or (op == "reduce" and d["acc"] == "f32")
        if r["status"] == "compile_error" or (r["status"] != "ok" and not unsafe):
            expect("lower " + ast, r, "ok")
print("KEVAL-TESTS", "OK" if not fail else "FAILED %d" % fail)
sys.exit(1 if fail else 0)
