"""可信评测 harness 的 C 源码（每个算子一份）。候选内核与它链接；候选拿不到参考结果（参考在内核调用之后才计算）。

协议：harness <seed> <mode:check|time> <min_reps> <min_seconds>
  - 每个形状两组独立随机输入（种子由评测器每次新取），交替调用，防止"缓存上一次输出"
  - 每次调用前输出缓冲区填 NaN；调用后核对输入未被改写（与副本逐字节比较）
  - 参考实现用 double 计算，并且在内核调用之后才计算
  - 打印一行 JSON
"""

COMMON = r"""
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
static uint64_t rs_;
static double urand(void) { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return (double)(rs_ >> 11) / 9007199254740992.0; }
static float* falloc(size_t n) { float* p = aligned_alloc(64, ((n * 4 + 63) / 64) * 64 + 64); return p; }
static void poison(float* p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = NAN; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }
static int check(const float* y, const double* ref, size_t n, double atol, double rtol, double* maxerr) {
  double me = 0; int bad = 0;
  for (size_t i = 0; i < n; i++) { double e = fabs((double)y[i] - ref[i]);
    if (!(e <= atol + rtol * fabs(ref[i]))) bad = 1; if (!(e <= me)) me = isnan(e) ? INFINITY : e; }
  *maxerr = me; return !bad; }
#define FAIL(fmt, ...) do { printf("{\"status\":\"" fmt "\"}\n", __VA_ARGS__); return 0; } while (0)
"""

# 每个算子：形状列表、输入分配/填充、调用、参考
OPS = {}

OPS["matmul"] = dict(shapes=[(256, 256, 256), (200, 312, 136)], body=r"""
void kernel(const float* restrict A, const float* restrict B, float* restrict C, int M, int N, int K);
static const int SH[][3] = {{256,256,256},{200,312,136}};
#define NSH 2
#define DECL int M = SH[s][0], N = SH[s][1], K = SH[s][2]; size_t nA = (size_t)M*K, nB = (size_t)K*N, nC = (size_t)M*N; \
  float *A[2], *B[2], *Ac[2], *Bc[2], *C = falloc(nC); double* ref = malloc(nC * sizeof(double)); \
  for (int t = 0; t < 2; t++) { A[t] = falloc(nA); B[t] = falloc(nB); Ac[t] = falloc(nA); Bc[t] = falloc(nB); \
    for (size_t i = 0; i < nA; i++) A[t][i] = (float)(2 * urand() - 1); for (size_t i = 0; i < nB; i++) B[t][i] = (float)(2 * urand() - 1); \
    memcpy(Ac[t], A[t], nA * 4); memcpy(Bc[t], B[t], nB * 4); }
#define CALL(t) kernel(A[t], B[t], C, M, N, K)
#define OUT C
#define NOUT nC
#define SAME(t) (!memcmp(A[t], Ac[t], nA * 4) && !memcmp(B[t], Bc[t], nB * 4))
#define REF(t) for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { double a = 0; for (int k = 0; k < K; k++) a += (double)A[t][(size_t)i*K+k] * B[t][(size_t)k*N+j]; ref[(size_t)i*N+j] = a; }
#define ATOL 2e-3
#define RTOL 1e-3
""")

OPS["softmax"] = dict(shapes=[(512, 1024), (300, 777)], body=r"""
void kernel(const float* restrict X, float* restrict Y, int R, int Cn);
static const int SH[][2] = {{512,1024},{300,777}};
#define NSH 2
#define DECL int R = SH[s][0], Cn = SH[s][1]; size_t n = (size_t)R*Cn; float *X[2], *Xc[2], *Y = falloc(n); double* ref = malloc(n * sizeof(double)); \
  for (int t = 0; t < 2; t++) { X[t] = falloc(n); Xc[t] = falloc(n); \
    for (int r = 0; r < R; r++) { double off = (r % 7 == 3) ? 80.0 : 0.0; for (int c = 0; c < Cn; c++) X[t][(size_t)r*Cn+c] = (float)(off + 20 * urand() - 10); } \
    memcpy(Xc[t], X[t], n * 4); }
#define CALL(t) kernel(X[t], Y, R, Cn)
#define OUT Y
#define NOUT n
#define SAME(t) (!memcmp(X[t], Xc[t], n * 4))
#define REF(t) for (int r = 0; r < R; r++) { const float* x = X[t] + (size_t)r*Cn; double m = -INFINITY, z = 0; \
    for (int c = 0; c < Cn; c++) m = x[c] > m ? x[c] : m; for (int c = 0; c < Cn; c++) z += exp(x[c] - m); \
    for (int c = 0; c < Cn; c++) ref[(size_t)r*Cn+c] = exp(x[c] - m) / z; }
#define ATOL 1e-7
#define RTOL 1e-4
""")

OPS["layernorm"] = dict(shapes=[(512, 1024), (300, 777)], body=r"""
void kernel(const float* restrict X, const float* restrict G, const float* restrict Bt, float* restrict Y, int R, int Cn, float eps);
static const int SH[][2] = {{512,1024},{300,777}};
#define NSH 2
#define DECL int R = SH[s][0], Cn = SH[s][1]; size_t n = (size_t)R*Cn; float *X[2], *Xc[2], *G = falloc(Cn), *Bt = falloc(Cn), *Gc = falloc(Cn), *Btc = falloc(Cn), \
  *Y = falloc(n); double* ref = malloc(n * sizeof(double)); float eps = 1e-5f; \
  for (int c = 0; c < Cn; c++) { G[c] = (float)(0.5 + urand()); Bt[c] = (float)(2 * urand() - 1); } memcpy(Gc, G, Cn * 4); memcpy(Btc, Bt, Cn * 4); \
  for (int t = 0; t < 2; t++) { X[t] = falloc(n); Xc[t] = falloc(n); \
    for (int r = 0; r < R; r++) { double off = (r % 2) ? 300.0 : 0.0; for (int c = 0; c < Cn; c++) X[t][(size_t)r*Cn+c] = (float)(off + 4 * urand() - 2); } \
    memcpy(Xc[t], X[t], n * 4); }
#define CALL(t) kernel(X[t], G, Bt, Y, R, Cn, eps)
#define OUT Y
#define NOUT n
#define SAME(t) (!memcmp(X[t], Xc[t], n * 4) && !memcmp(G, Gc, Cn * 4) && !memcmp(Bt, Btc, Cn * 4))
#define REF(t) for (int r = 0; r < R; r++) { const float* x = X[t] + (size_t)r*Cn; double m = 0, v = 0; \
    for (int c = 0; c < Cn; c++) m += x[c]; m /= Cn; for (int c = 0; c < Cn; c++) v += (x[c] - m) * (x[c] - m); v /= Cn; \
    for (int c = 0; c < Cn; c++) ref[(size_t)r*Cn+c] = (x[c] - m) / sqrt(v + (double)eps) * G[c] + Bt[c]; }
#define ATOL 1e-3
#define RTOL 1e-3
""")

OPS["reduce"] = dict(shapes=[(1 << 23,), (1000003,)], body=r"""
void kernel(const float* restrict X, float* restrict out, long n);
static const long SH[][1] = {{1L << 23},{1000003L}};
#define NSH 2
#define DECL long n = SH[s][0]; float *X[2], *Xc[2], *Y = falloc(1); double* ref = malloc(sizeof(double)); \
  for (int t = 0; t < 2; t++) { X[t] = falloc(n); Xc[t] = falloc(n); for (long i = 0; i < n; i++) X[t][i] = (float)urand(); memcpy(Xc[t], X[t], n * 4); }
#define CALL(t) kernel(X[t], Y, n)
#define OUT Y
#define NOUT 1
#define SAME(t) (!memcmp(X[t], Xc[t], n * 4))
#define REF(t) { double a = 0; for (long i = 0; i < n; i++) a += X[t][i]; ref[0] = a; }
#define ATOL 0
#define RTOL 1e-6
""")

MAIN = r"""
int main(int argc, char** argv) {
  rs_ = strtoull(argv[1], 0, 10) | 1; int timing = !strcmp(argv[2], "time"); int minreps = atoi(argv[3]); double mins = atof(argv[4]);
  printf("{\"status\":\"ok\",\"shapes\":[");
  for (int s = 0; s < NSH; s++) {
    DECL
    double me[2];
    for (int t = 0; t < 2; t++) {
      poison(OUT, NOUT); CALL(t);
      if (!SAME(t)) { printf("]}\n{\"status\":\"hack\",\"why\":\"input-modified\",\"shape\":%d}\n", s); return 0; }
      REF(t)
      if (!check(OUT, ref, NOUT, ATOL, RTOL, &me[t])) { printf("]}\n{\"status\":\"incorrect\",\"shape\":%d,\"max_err\":%g}\n", s, me[t] > 1e300 ? 1e300 : me[t]); return 0; }
    }
    double tm = 0, tmin = 0; int reps = 0;
    if (timing) {
      for (int w = 0; w < 2; w++) { poison(OUT, NOUT); CALL(w & 1); }
      double* ts = malloc(sizeof(double) * 100000); double t0 = now();
      while ((reps < minreps || now() - t0 < mins) && reps < 100000) { poison(OUT, 1); double a = now(); CALL(reps & 1); ts[reps++] = now() - a; }
      qsort(ts, reps, sizeof(double), cmpd); tm = ts[reps / 2] * 1e3; tmin = ts[0] * 1e3;
      int last = (reps - 1) & 1; REF(last)
      if (!check(OUT, ref, NOUT, ATOL, RTOL, &me[0])) { printf("]}\n{\"status\":\"incorrect\",\"why\":\"timed-output\",\"shape\":%d}\n", s); return 0; }
    }
    printf("%s{\"ms\":%.6f,\"ms_min\":%.6f,\"reps\":%d,\"max_err\":%g}", s ? "," : "", tm, tmin, reps, me[0] > me[1] ? me[0] : me[1]);
  }
  printf("]}\n"); return 0;
}
"""


def source(op):
    return COMMON + OPS[op]["body"] + MAIN
