#!/usr/bin/env python3
"""隔离评测：静态检查 → 编译 → 符号检查 → 正确性门 → 计时。候选（无论来自 Aura lowering 还是 LLM 文本）走同一条路。

  keval.py eval --op OP --src cand.c --out res.json [--no-time]
  keval.py baseline --op OP --out res.json     # 朴素源码在 -O3（开自动向量化）下的"编译器基线"

隔离与防奖励黑客（对应 SOL-ExecBench / 文章里的评测区隔离）：
  1) 静态：#include 白名单；禁止 asm、非 inline 的 static、constructor、optimize/target 属性、系统/时间/进程/文件调用、引用 harness 符号
  2) 编译参数固定（-O2 -march=native -fno-tree-vectorize），候选不能改
  3) 符号：候选目标文件只能导出 `kernel`；不能有可写全局数据（.data/.bss）；未定义符号只能是白名单里的 libm/libc 函数
  4) 动态：每次新随机种子、两组输入交替、输出预先填 NaN、输入防改写、参考在内核调用之后才计算、多形状（含非 2 次幂）
  5) 在全新临时目录里编译运行；子进程有 CPU/内存/时间上限；计时段用 flock 串行并 taskset 绑核（公平队列，防资源争抢）
"""
import argparse, hashlib, json, os, re, resource, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness  # noqa: E402

CC = os.environ.get("AKB_CC", "gcc")
CFLAGS = ["-std=gnu11", "-O2", "-march=native", "-fno-tree-vectorize", "-fno-unroll-loops", "-fno-strict-aliasing"]
CORE = os.environ.get("AKB_CORE", "7")
LOCK = os.environ.get("AKB_LOCK", "/tmp/akb-time.lock")
INCLUDES = {"math.h", "string.h", "stddef.h", "stdint.h", "stdlib.h", "immintrin.h", "float.h", "limits.h", "stdbool.h"}
BANNED = [
    (r"\b(asm|__asm__|__asm)\b", "inline-asm"),
    (r"\bstatic\b(?!\s+(inline|const)\b)(?!\s+__inline)", "static-storage"),
    (r"\b(constructor|destructor)\b", "ctor-dtor"),
    (r"\b(optimize|target)\s*\(", "flag-override"),
    (r"#\s*pragma\s+GCC\s+(optimize|target|push_options)", "flag-override"),
    (r"\b(system|popen|fopen|freopen|open|openat|read|write|getenv|setenv|dlopen|dlsym|syscall|fork|vfork|exec\w*|exit|_exit|abort|"
     r"signal|sigaction|setjmp|longjmp|mmap|mprotect|pthread_\w+|clock|clock_gettime|gettimeofday|time|__rdtsc|rdtsc|sleep|usleep|"
     r"printf|fprintf|puts|scanf)\s*\(", "forbidden-call"),
    (r"\b(main|urand|rs_|poison|check|now|falloc|ref)\b", "harness-symbol"),
    (r"__builtin_(frame_address|return_address)", "stack-peek"),
]
UNDEF_OK = {"expf", "exp", "sqrtf", "sqrt", "fmaxf", "fminf", "fabsf", "fabs", "logf", "log", "tanhf", "erff", "powf", "exp2f",
            "memcpy", "memset", "memmove", "malloc", "free", "aligned_alloc", "posix_memalign", "calloc", "__stack_chk_fail",
            "_GLOBAL_OFFSET_TABLE_"}


def static_check(src):
    flags = []
    for m in re.finditer(r'#\s*include\s*[<"]([^>"]+)[>"]', src):
        if m.group(1) not in INCLUDES:
            flags.append("include:" + m.group(1))
    code = re.sub(r"//[^\n]*|/\*.*?\*/", " ", src, flags=re.S)
    code = re.sub(r'"(\\.|[^"\\])*"', '""', code)
    for pat, why in BANNED:
        if re.search(pat, code):
            flags.append(why)
    if not re.search(r"\bvoid\s+kernel\s*\(", code):
        flags.append("no-kernel")
    return flags


def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def sh(cmd, cwd, timeout, pre=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, preexec_fn=pre, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")})
        return p.returncode, p.stdout, p.stderr, time.time() - t0
    except subprocess.TimeoutExpired:
        return -999, "", "timeout", time.time() - t0


def harness_obj(op, d):
    hsrc = harness.source(op); h = hashlib.sha256((hsrc + " ".join(CFLAGS)).encode()).hexdigest()[:16]
    cache = os.path.join(tempfile.gettempdir(), "akb-harness-%s-%s.o" % (op, h))
    if not os.path.exists(cache):
        open(os.path.join(d, "h.c"), "w").write(hsrc)
        rc, _, err, _ = sh([CC] + CFLAGS + ["-c", "h.c", "-o", "h.o"], d, 120)
        if rc != 0:
            raise RuntimeError("harness 编译失败: " + err[:500])
        tmp = cache + ".%d" % os.getpid(); shutil.copy(os.path.join(d, "h.o"), tmp); os.replace(tmp, cache)
    return cache


def symbol_check(d):
    rc, out, _, _ = sh(["nm", "cand.o"], d, 30)
    flags = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("U", "w"):
            if parts[1] not in UNDEF_OK and not re.fullmatch(r"_ZGV\w+_(expf|exp|logf|log|sqrtf|tanhf|erff|powf|exp2f)", parts[1]):
                flags.append("undef:" + parts[1])
        elif len(parts) == 3:
            t, name = parts[1], parts[2]
            if t in "dDbBcCgGsS":
                flags.append("mutable-global:" + name)
            elif t in "TW" and name != "kernel":
                flags.append("extra-export:" + name)
    return flags


def evaluate(op, src, do_time=True, cflags=None, skip_static=False, min_reps=5, min_s=0.15):
    res = {"op": op, "src_sha256": hashlib.sha256(src.encode()).hexdigest()}
    t_all = time.time()
    flags = [] if skip_static else static_check(src)
    if flags:
        res.update(status="hack", flags=flags); return res
    d = tempfile.mkdtemp(prefix="akb-")
    try:
        open(os.path.join(d, "cand.c"), "w").write(src)
        rc, _, err, ct = sh([CC] + (cflags or CFLAGS) + ["-c", "cand.c", "-o", "cand.o"], d, 120)
        res["compile_s"] = round(ct, 3)
        if rc != 0:
            res.update(status="compile_error", error=err[-1500:]); return res
        flags = [] if skip_static else symbol_check(d)
        if flags:
            res.update(status="hack", flags=flags); return res
        ho = harness_obj(op, d)
        rc, _, err, _ = sh([CC, "cand.o", ho, "-o", "run", "-lm"], d, 60)
        if rc != 0:
            res.update(status="compile_error", error="link: " + err[-1500:]); return res
        seed = str(int.from_bytes(os.urandom(6), "little"))
        rc, out, err, _ = sh(["./run", seed, "check", "1", "0"], d, 120, pre=limits)
        r = parse_out(rc, out, err)
        if r["status"] != "ok" or not do_time:
            res.update(r); return res
        seed = str(int.from_bytes(os.urandom(6), "little"))
        cmd = ["flock", LOCK, "taskset", "-c", CORE, "./run", seed, "time", str(min_reps), str(min_s)]
        rc, out, err, _ = sh(cmd, d, 600, pre=limits)
        r = parse_out(rc, out, err); res.update(r)
        if r["status"] == "ok":
            ms = [s["ms"] for s in r["shapes"]]
            g = 1.0
            for m in ms:
                g *= m
            res["ms_geo"] = g ** (1.0 / len(ms))
        return res
    finally:
        res["eval_s"] = round(time.time() - t_all, 3)
        shutil.rmtree(d, ignore_errors=True)


def parse_out(rc, out, err):
    if rc == -999:
        return {"status": "timeout"}
    if rc != 0:
        return {"status": "crash", "rc": rc, "error": err[-500:]}
    try:
        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        return {"status": "crash", "error": "bad harness output: " + out[-300:]}


NAIVE = {   # 朴素 C 源码（"编译器基线"用：同一源码交给 -O3 自动向量化）
    "matmul": "void kernel(const float* restrict A, const float* restrict B, float* restrict C, int M, int N, int K) {\n"
              "  for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { float s = 0; for (int k = 0; k < K; k++) s += A[(long)i*K+k] * B[(long)k*N+j]; C[(long)i*N+j] = s; } }\n",
    "softmax": "#include <math.h>\nvoid kernel(const float* restrict X, float* restrict Y, int R, int Cn) {\n"
               "  for (int r = 0; r < R; r++) { const float* x = X + (long)r*Cn; float* y = Y + (long)r*Cn; float m = x[0];\n"
               "    for (int c = 1; c < Cn; c++) m = x[c] > m ? x[c] : m; float s = 0;\n"
               "    for (int c = 0; c < Cn; c++) { y[c] = expf(x[c] - m); s += y[c]; } for (int c = 0; c < Cn; c++) y[c] /= s; } }\n",
    "layernorm": "#include <math.h>\nvoid kernel(const float* restrict X, const float* restrict G, const float* restrict Bt, float* restrict Y, int R, int Cn, float eps) {\n"
                 "  for (int r = 0; r < R; r++) { const float* x = X + (long)r*Cn; float* y = Y + (long)r*Cn; float m = 0, v = 0;\n"
                 "    for (int c = 0; c < Cn; c++) m += x[c]; m /= Cn; for (int c = 0; c < Cn; c++) v += (x[c]-m)*(x[c]-m); v /= Cn;\n"
                 "    float rs = 1.0f / sqrtf(v + eps); for (int c = 0; c < Cn; c++) y[c] = (x[c]-m)*rs*G[c] + Bt[c]; } }\n",
    "reduce": "void kernel(const float* restrict X, float* restrict out, long n) { double s = 0; for (long i = 0; i < n; i++) s += X[i]; out[0] = (float)s; }\n",
}


def main():
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("eval"); s.add_argument("--op", required=True); s.add_argument("--src", required=True)
    s.add_argument("--out", required=True); s.add_argument("--no-time", action="store_true")
    s = sp.add_parser("baseline"); s.add_argument("--op", required=True); s.add_argument("--out", required=True)
    s.add_argument("--flags", default="-std=gnu11 -O3 -march=native -ffast-math")
    a = p.parse_args()
    if a.cmd == "eval":
        r = evaluate(a.op, open(a.src).read(), do_time=not a.no_time)
    else:
        r = evaluate(a.op, NAIVE[a.op], cflags=a.flags.split()); r["cflags"] = a.flags
    tmp = a.out + ".tmp"; json.dump(r, open(tmp, "w"), ensure_ascii=False); os.replace(tmp, a.out)
    print(json.dumps({k: r.get(k) for k in ("status", "ms_geo", "flags", "eval_s")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
