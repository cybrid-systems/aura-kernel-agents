#!/usr/bin/env python3
"""单核"光速"（SOL）近似：实测峰值 FLOP/s（AVX-512 FMA，12 路独立累加）与给定工作集大小下的读带宽，
按 roofline 给每个算子形状估 T_SOL = max(flops/峰值, 字节/带宽)。这只是 SOL-ExecBench（B200 + SOLAR）思路在 CPU 上的类比。
  calib.py --out results/calib.json
"""
import argparse, json, os, subprocess, tempfile

SRC = r"""
#define _POSIX_C_SOURCE 199309L
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char** argv) {
  if (argv[1][0] == 'f') {
    __m512 m = _mm512_set1_ps(0.999999f), c = _mm512_set1_ps(1e-7f);
#define V(i) __m512 a##i = _mm512_set1_ps((float)i);
    V(0) V(1) V(2) V(3) V(4) V(5) V(6) V(7) V(8) V(9) V(10) V(11)
    long it = 20000000; double best = 1e9;
    for (int r = 0; r < 5; r++) { double t0 = now();
      for (long i = 0; i < it; i++) {
#define F(i) a##i = _mm512_fmadd_ps(a##i, m, c);
        F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7) F(8) F(9) F(10) F(11) }
      double t = now() - t0; if (t < best) best = t; }
    float s = _mm512_reduce_add_ps(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11);
    printf("{\"gflops\":%.3f,\"chk\":%g}\n", it * 12.0 * 16 * 2 / best / 1e9, s);
  } else {
    long n = atol(argv[2]) / 4; n = (n / 128) * 128; float* x = aligned_alloc(64, n * 4); for (long i = 0; i < n; i++) x[i] = 1.0f;
    __m512 acc[8]; for (int j = 0; j < 8; j++) acc[j] = _mm512_setzero_ps();
    double best = 1e9; int reps = (int)(2e9 / (n * 4.0)) + 3;
    for (int r = 0; r < reps; r++) { double t0 = now();
      for (long i = 0; i < n; i += 128) for (int j = 0; j < 8; j++) acc[j] = _mm512_add_ps(acc[j], _mm512_load_ps(x + i + 16 * j));
      double t = now() - t0; if (t < best) best = t; }
    float s = 0; for (int j = 0; j < 8; j++) s += _mm512_reduce_add_ps(acc[j]);
    printf("{\"gbps\":%.3f,\"chk\":%g}\n", n * 4.0 / best / 1e9, s);
  }
  return 0;
}
"""
SHAPES = {"matmul": [(256, 256, 256), (200, 312, 136)], "softmax": [(512, 1024), (300, 777)],
          "layernorm": [(512, 1024), (300, 777)], "reduce": [(1 << 23,), (1000003,)]}
EXP_FLOPS = 10   # 假设：一次 exp 计 10 flop


def work(op, sh):
    if op == "matmul":
        M, N, K = sh; return 2.0 * M * N * K, 4.0 * (M * K + K * N + M * N)
    if op == "softmax":
        R, C = sh; return R * C * (EXP_FLOPS + 3.0), 8.0 * R * C
    if op == "layernorm":
        R, C = sh; return 7.0 * R * C, 8.0 * R * C + 8.0 * C
    (n,) = sh; return float(n), 4.0 * n


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True); a = p.parse_args()
    core = os.environ.get("AKB_CORE", "7"); d = tempfile.mkdtemp()
    open(d + "/c.c", "w").write(SRC)
    subprocess.check_call(["gcc", "-O2", "-march=native", "c.c", "-o", "c"], cwd=d)
    run = lambda *args: json.loads(subprocess.check_output(["flock", "/tmp/akb-time.lock", "taskset", "-c", core, "./c"] + list(args), cwd=d))
    peak = run("f")["gflops"]; out = {"peak_gflops_1core": peak, "exp_flops_assumed": EXP_FLOPS, "core": core, "ops": {}}
    for op, shapes in SHAPES.items():
        rows = []; g = 1.0
        for sh in shapes:
            fl, by = work(op, sh); bw = run("b", str(int(by)))["gbps"]
            t = max(fl / (peak * 1e9), by / (bw * 1e9)) * 1e3; g *= t
            rows.append({"shape": sh, "flops": fl, "bytes": by, "read_gbps_at_ws": bw, "t_sol_ms": t})
        out["ops"][op] = {"shapes": rows, "t_sol_ms_geo": g ** (1.0 / len(shapes))}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps({op: round(v["t_sol_ms_geo"], 5) for op, v in out["ops"].items()}), "peak", peak)


if __name__ == "__main__":
    main()
