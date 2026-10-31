#!/usr/bin/env python3
"""调度 AST（S 表达式）→ C 源码的 lowering。

Aura 侧持有的"内核程序"是一棵 S 表达式，例如
    (matmul (order i k j) (tile-i 64) (tile-j 256) (tile-k 128) (unroll 2) (vec 16))
算法（算子语义）固定，AST 只描述调度（Halide 风格）。本工具把它 lowering 成一个 C 函数 `kernel`，
签名见 SIGS。生成的代码只用 GCC 向量扩展，不依赖编译器自动向量化（评测固定 -fno-tree-vectorize）。

  kgen.py space                    -> 打印 JSON：各算子的旋钮、合法取值、起点（朴素实现）
  kgen.py lower --ast "(...)" --out f.c
"""
import argparse, json, re, sys

SIGS = {
    "matmul": "void kernel(const float* restrict A, const float* restrict B, float* restrict C, int M, int N, int K)",
    "softmax": "void kernel(const float* restrict X, float* restrict Y, int R, int Cn)",
    "layernorm": "void kernel(const float* restrict X, const float* restrict G, const float* restrict Bt, float* restrict Y, int R, int Cn, float eps)",
    "reduce": "void kernel(const float* restrict X, float* restrict out, long n)",
}
SEMANTICS = {
    "matmul": "C[M,N] = A[M,K] @ B[K,N]，行主序 fp32，C 被覆盖写",
    "softmax": "Y[r,:] = softmax(X[r,:])，逐行，fp32",
    "layernorm": "Y[r,:] = (X[r,:]-mean)/sqrt(var+eps)*G + Bt，var 为总体方差，逐行，fp32",
    "reduce": "out[0] = sum(X[0..n))，要求与 double 参考的相对误差 ≤ 1e-6",
}
SPACE = {
    "matmul": {"order": ["ijk", "ikj", "jik", "jki", "kij", "kji"], "tile-i": [0, 8, 16, 32, 64, 128],
               "tile-j": [0, 32, 64, 128, 256, 512], "tile-k": [0, 16, 32, 64, 128, 256],
               "unroll": [1, 2, 4, 8], "vec": [1, 4, 8, 16], "micro": [0, 2, 4, 6, 8]},
    "softmax": {"passes": [3, 2], "exp": ["libm", "poly"], "unroll": [1, 2, 4], "vec": [1, 4, 8, 16]},
    "layernorm": {"stats": ["two-pass", "sumsq"], "acc": ["f32", "f64"], "unroll": [1, 2, 4], "vec": [1, 4, 8, 16]},
    "reduce": {"acc": ["f32", "f64"], "unroll": [1, 2, 4, 8], "vec": [1, 4, 8, 16]},
}
START = {
    "matmul": {"order": "ijk", "tile-i": 0, "tile-j": 0, "tile-k": 0, "unroll": 1, "vec": 1, "micro": 0},
    "softmax": {"passes": 3, "exp": "libm", "unroll": 1, "vec": 1},
    "layernorm": {"stats": "two-pass", "acc": "f32", "unroll": 1, "vec": 1},
    "reduce": {"acc": "f64", "unroll": 1, "vec": 1},
}


def parse_sexpr(s):
    toks = re.findall(r"\(|\)|[^\s()]+", s)
    pos = 0
    def rd():
        nonlocal pos
        t = toks[pos]; pos += 1
        if t == "(":
            out = []
            while toks[pos] != ")":
                out.append(rd())
            pos += 1
            return out
        return int(t) if re.fullmatch(r"-?\d+", t) else t
    v = rd()
    if pos != len(toks):
        raise ValueError("多余的 token")
    return v


def ast_to_sched(ast):
    """(op (knob v) ...) -> (op, dict)。order 写成 (order i k j)。非法旋钮/取值抛 ValueError。"""
    if not isinstance(ast, list) or not ast or ast[0] not in SPACE:
        raise ValueError("未知算子: %r" % (ast[:1],))
    op = ast[0]; d = dict(START[op])
    for node in ast[1:]:
        if not isinstance(node, list) or len(node) < 2:
            raise ValueError("坏节点: %r" % (node,))
        k = node[0]; v = "".join(node[1:]) if k == "order" else node[1]
        if k not in SPACE[op]:
            raise ValueError("未知旋钮 %s" % k)
        if v not in SPACE[op][k]:
            raise ValueError("非法取值 %s=%r" % (k, v))
        d[k] = v
    return op, d


def sched_to_ast(op, d):
    parts = [op]
    for k in SPACE[op]:
        v = d[k]
        parts.append("(order %s)" % " ".join(v) if k == "order" else "(%s %s)" % (k, v))
    return "(" + " ".join(parts) + ")"


def prelude(W, need_exp=False, need_f64=False):
    s = ["#include <math.h>", "#include <string.h>", "#include <stddef.h>",
         "typedef float vf __attribute__((vector_size(%d), aligned(4)));" % (4 * W),
         "typedef int vi __attribute__((vector_size(%d), aligned(4)));" % (4 * W),
         "static inline vf ld(const float* p) { return *(const vf*)p; }   /* vf 带 aligned(4)：生成非对齐向量访存 */",
         "static inline void st(float* p, vf v) { *(vf*)p = v; }",
         "static inline vf splat(float x) { vf v = {0}; return v + x; }",
         "static inline vf vmax(vf a, vf b) { vi m = a > b; return (vf)(((vi)a & m) | ((vi)b & ~m)); }",
         "static inline float hsum(vf v) { float s = 0; for (int l = 0; l < %d; l++) s += v[l]; return s; }" % W,
         "static inline float hmax(vf v) { float s = v[0]; for (int l = 1; l < %d; l++) s = v[l] > s ? v[l] : s; return s; }" % W]
    if need_f64:
        s += ["typedef double vd __attribute__((vector_size(%d), aligned(8)));" % (8 * W),
              "static inline vd cvt(vf v) { return __builtin_convertvector(v, vd); }",
              "static inline double hsumd(vd v) { double s = 0; for (int l = 0; l < %d; l++) s += v[l]; return s; }" % W]
    if need_exp:   # 2^n * p(f) 的多项式 exp，|相对误差| ~ 2e-7，输入钳位到 [-87, 88]
        s += ["static inline vf vexp(vf x) {",
              "  x = vmax(x, splat(-87.0f)); x = -vmax(-x, splat(-88.0f));",
              "  vf t = x * 1.44269504088896341f;",
              "  vi n = __builtin_convertvector(t, vi); vf fn = __builtin_convertvector(n, vf);",
              "  vi adj = fn > t; n = n + adj; fn = __builtin_convertvector(n, vf);",
              "  vf f = (x - fn * 0.693359375f) - fn * -2.12194440e-4f;",
              "  vf p = splat(1.9875691500e-4f); p = p * f + 1.3981999507e-3f; p = p * f + 8.3334519073e-3f;",
              "  p = p * f + 4.1665795894e-2f; p = p * f + 1.6666665459e-1f; p = p * f + 5.0000001201e-1f;",
              "  p = p * f * f + f + 1.0f;",
              "  vi e = (n + 127) << 23; return p * (vf)e; }"]
    return s


# ---------------- matmul ----------------
def lower_matmul(d):
    o = list(d["order"]); W = d["vec"]; U = d["unroll"]
    T = {"i": d["tile-i"], "j": d["tile-j"], "k": d["tile-k"]}; D = {"i": "M", "j": "N", "k": "K"}
    L = prelude(W) + [SIGS["matmul"] + " {", "  memset(C, 0, sizeof(float) * (size_t)M * N);"]
    ind = 1; closes = 0
    def emit(x):
        L.append("  " * ind + x)
    for v in ("i", "k", "j"):     # 分块循环顺序固定 i,k,j
        if T[v]:
            emit("for (int %s0 = 0; %s0 < %s; %s0 += %d) { int %s1 = %s0 + %d < %s ? %s0 + %d : %s;"
                 % (v, v, D[v], v, T[v], v, v, T[v], D[v], v, T[v], D[v]))
        else:
            emit("{ int %s0 = 0, %s1 = %s;" % (v, v, D[v]))
        ind += 1; closes += 1
    if d["micro"] and W > 1:   # 寄存器分块微内核：RB 行 × (W*U) 列的累加器常驻寄存器，k 最内；忽略 order
        RB = d["micro"]
        emit("int i = i0;")
        emit("for (; i + %d <= i1; i += %d) {" % (RB, RB)); ind += 1
        emit("int j = j0;")
        emit("for (; j + %d <= j1; j += %d) {" % (W * U, W * U)); ind += 1
        emit(" ".join("vf c%d_%d = splat(0.0f);" % (r, u) for r in range(RB) for u in range(U)))
        emit("for (int k = k0; k < k1; k++) {"); ind += 1
        emit("const float* restrict b = B + (size_t)k * N + j; " + " ".join("vf b%d = ld(b + %d);" % (u, u * W) for u in range(U)))
        for r in range(RB):
            emit("{ const float a = A[(size_t)(i + %d) * K + k]; %s }" % (r, " ".join("c%d_%d += a * b%d;" % (r, u, u) for u in range(U))))
        ind -= 1; emit("}")
        for r in range(RB):
            emit(" ".join("st(C + (size_t)(i + %d) * N + j + %d, ld(C + (size_t)(i + %d) * N + j + %d) + c%d_%d);" % (r, u * W, r, u * W, r, u) for u in range(U)))
        ind -= 1; emit("}")
        emit("for (; j < j1; j++) for (int r = 0; r < %d; r++) { float s = 0; for (int k = k0; k < k1; k++) s += A[(size_t)(i + r) * K + k] * B[(size_t)k * N + j]; C[(size_t)(i + r) * N + j] += s; }" % RB)
        ind -= 1; emit("}")
        emit("for (; i < i1; i++) for (int k = k0; k < k1; k++) { const float a = A[(size_t)i * K + k]; int j = j0;")
        emit("  for (; j + %d <= j1; j += %d) st(C + (size_t)i * N + j, ld(C + (size_t)i * N + j) + a * ld(B + (size_t)k * N + j));" % (W, W))
        emit("  for (; j < j1; j++) C[(size_t)i * N + j] += a * B[(size_t)k * N + j]; }")
        for _ in range(closes):
            ind -= 1; emit("}")
        L.append("}")
        return L
    for v in o[:2]:
        emit("for (int %s = %s0; %s < %s1; %s++) {" % (v, v, v, v, v)); ind += 1; closes += 1
    x = o[2]
    if x == "j":
        emit("const float a = A[(size_t)i * K + k]; float* restrict c = C + (size_t)i * N; const float* restrict b = B + (size_t)k * N;")
        emit("int j = j0;")
        if W > 1:
            body = " ".join("st(c + j + %d, ld(c + j + %d) + a * ld(b + j + %d));" % (u * W, u * W, u * W) for u in range(U))
            emit("for (; j + %d <= j1; j += %d) { %s }" % (W * U, W * U, body))
            if U > 1:
                emit("for (; j + %d <= j1; j += %d) st(c + j, ld(c + j) + a * ld(b + j));" % (W, W))
        elif U > 1:
            body = " ".join("c[j + %d] += a * b[j + %d];" % (u, u) for u in range(U))
            emit("for (; j + %d <= j1; j += %d) { %s }" % (U, U, body))
        emit("for (; j < j1; j++) c[j] += a * b[j];")
    elif x == "k":   # 点积：U 个独立累加器
        emit("const float* restrict a = A + (size_t)i * K; float s[%d] = {0}; int k = k0;" % U)
        body = " ".join("s[%d] += a[k + %d] * B[(size_t)(k + %d) * N + j];" % (u, u, u) for u in range(U))
        emit("for (; k + %d <= k1; k += %d) { %s }" % (U, U, body))
        emit("for (; k < k1; k++) s[0] += a[k] * B[(size_t)k * N + j];")
        emit("float t = 0; for (int u = 0; u < %d; u++) t += s[u]; C[(size_t)i * N + j] += t;" % U)
    else:            # i 最内：按列跨步
        emit("const float b = B[(size_t)k * N + j]; int i = i0;")
        body = " ".join("C[(size_t)(i + %d) * N + j] += A[(size_t)(i + %d) * K + k] * b;" % (u, u) for u in range(U))
        emit("for (; i + %d <= i1; i += %d) { %s }" % (U, U, body))
        emit("for (; i < i1; i++) C[(size_t)i * N + j] += A[(size_t)i * K + k] * b;")
    for _ in range(closes):
        ind -= 1; emit("}")
    L.append("}")
    return L


# ---------------- softmax ----------------
def lower_softmax(d):
    W = d["vec"]; U = d["unroll"]; poly = d["exp"] == "poly"; S = W * U
    ex = "vexp(%s)" if poly else "lexp(%s)"
    L = prelude(W, need_exp=poly)
    if not poly:
        L.append("static inline vf lexp(vf x) { vf r; for (int l = 0; l < %d; l++) r[l] = expf(x[l]); return r; }" % W)
    L += [SIGS["softmax"] + " {", "  for (int r = 0; r < R; r++) {",
          "    const float* restrict x = X + (size_t)r * Cn; float* restrict y = Y + (size_t)r * Cn; int c;"]
    acc = lambda name, init: " ".join("vf %s%d = splat(%s);" % (name, u, init) for u in range(U))
    if d["passes"] == 3:
        L.append("    %s" % acc("m", "-INFINITY"))
        L.append("    for (c = 0; c + %d <= Cn; c += %d) { %s }" % (S, S, " ".join("m%d = vmax(m%d, ld(x + c + %d));" % (u, u, u * W) for u in range(U))))
        L.append("    vf mm = m0; %s float m = hmax(mm); for (; c < Cn; c++) m = x[c] > m ? x[c] : m;" % " ".join("mm = vmax(mm, m%d);" % u for u in range(1, U)))
        L.append("    %s" % acc("s", "0.0f"))
        L.append("    for (c = 0; c + %d <= Cn; c += %d) { %s }" % (S, S, " ".join(
            "{ vf e = %s; st(y + c + %d, e); s%d += e; }" % (ex % ("(ld(x + c + %d) - m)" % (u * W)), u * W, u) for u in range(U))))
        L.append("    float s = 0; %s for (; c < Cn; c++) { float e = expf(x[c] - m); y[c] = e; s += e; }" % " ".join("s += hsum(s%d);" % u for u in range(U)))
        L.append("    float inv = 1.0f / s;")
        L.append("    for (c = 0; c + %d <= Cn; c += %d) { %s }" % (W, W, "st(y + c, ld(y + c) * inv);"))
        L.append("    for (; c < Cn; c++) y[c] *= inv;")
    else:   # 在线 max/sum（逐 lane），第二遍重算 exp 并归一化
        L.append("    %s %s" % (acc("m", "-INFINITY"), acc("s", "0.0f")))
        L.append("    for (c = 0; c + %d <= Cn; c += %d) { %s }" % (S, S, " ".join(
            "{ vf v = ld(x + c + %d); vf nm = vmax(m%d, v); s%d = s%d * %s + %s; m%d = nm; }"
            % (u * W, u, u, u, ex % ("(m%d - nm)" % u), ex % "(v - nm)", u) for u in range(U))))
        L.append("    float m = -INFINITY; %s" % " ".join("m = hmax(m%d) > m ? hmax(m%d) : m;" % (u, u) for u in range(U)))
        L.append("    for (int cc = c; cc < Cn; cc++) m = x[cc] > m ? x[cc] : m;")
        L.append("    float s = 0; %s" % " ".join("s += hsum(s%d * %s);" % (u, ex % ("(m%d - m)" % u)) for u in range(U)))
        L.append("    for (int cc = c; cc < Cn; cc++) s += expf(x[cc] - m);")
        L.append("    float inv = 1.0f / s;")
        L.append("    for (c = 0; c + %d <= Cn; c += %d) st(y + c, %s * inv);" % (W, W, ex % "(ld(x + c) - m)"))
        L.append("    for (; c < Cn; c++) y[c] = expf(x[c] - m) * inv;")
    L += ["  }", "}"]
    return L


# ---------------- layernorm ----------------
def lower_layernorm(d):
    W = d["vec"]; U = d["unroll"]; S = W * U; f64 = d["acc"] == "f64"
    L = prelude(W, need_f64=f64)
    T, Z, CV, HS = ("vd", "(vd){0}", "cvt", "hsumd") if f64 else ("vf", "splat(0.0f)", "", "hsum")
    sc = "double" if f64 else "float"
    L += [SIGS["layernorm"] + " {", "  for (int r = 0; r < R; r++) {",
          "    const float* restrict x = X + (size_t)r * Cn; float* restrict y = Y + (size_t)r * Cn; int c;"]
    decl = lambda n: " ".join("%s %s%d = %s;" % (T, n, u, Z) for u in range(U))
    if d["stats"] == "two-pass":
        L.append("    %s" % decl("s"))
        L.append("    for (c = 0; c + %d <= Cn; c += %d) { %s }" % (S, S, " ".join("s%d += %s(ld(x + c + %d));" % (u, CV, u * W) for u in range(U))))
        L.append("    %s sum = 0; %s for (; c < Cn; c++) sum += x[c];" % (sc, " ".join("sum += %s(s%d);" % (HS, u) for u in range(U))))
        L.append("    float mean = (float)(sum / Cn);")
        L.append("    %s" % decl("q"))
        L.append("    for (c = 0; c + %d <= Cn; c += %d) { %s }" % (S, S, " ".join(
            "{ vf t = ld(x + c + %d) - mean; q%d += %s(t * t); }" % (u * W, u, CV) for u in range(U))))
        L.append("    %s sq = 0; %s for (; c < Cn; c++) { float t = x[c] - mean; sq += t * t; }" % (sc, " ".join("sq += %s(q%d);" % (HS, u) for u in range(U))))
        L.append("    float var = (float)(sq / Cn);")
    else:
        L.append("    %s %s" % (decl("s"), decl("q")))
        L.append("    for (c = 0; c + %d <= Cn; c += %d) { %s }" % (S, S, " ".join(
            "{ %s v = %s(ld(x + c + %d)); s%d += v; q%d += v * v; }" % (T, CV, u * W, u, u) for u in range(U))))
        L.append("    %s sum = 0, sq = 0; %s for (; c < Cn; c++) { %s v = x[c]; sum += v; sq += v * v; }" % (
            sc, " ".join("sum += %s(s%d); sq += %s(q%d);" % (HS, u, HS, u) for u in range(U)), sc))
        L.append("    %s mu = sum / Cn; float mean = (float)mu; float var = (float)(sq / Cn - mu * mu); if (var < 0) var = 0;" % sc)
    L.append("    float rs = 1.0f / sqrtf(var + eps);")
    L.append("    for (c = 0; c + %d <= Cn; c += %d) st(y + c, (ld(x + c) - mean) * rs * ld(G + c) + ld(Bt + c));" % (W, W))
    L.append("    for (; c < Cn; c++) y[c] = (x[c] - mean) * rs * G[c] + Bt[c];")
    L += ["  }", "}"]
    return L


# ---------------- reduce ----------------
def lower_reduce(d):
    W = d["vec"]; U = d["unroll"]; S = W * U; f64 = d["acc"] == "f64"
    L = prelude(W, need_f64=f64)
    T, Z, CV, HS, sc = ("vd", "(vd){0}", "cvt", "hsumd", "double") if f64 else ("vf", "splat(0.0f)", "", "hsum", "float")
    L += [SIGS["reduce"] + " {",
          "  %s long i = 0;" % " ".join("%s a%d = %s;" % (T, u, Z) for u in range(U)),
          "  for (; i + %d <= n; i += %d) { %s }" % (S, S, " ".join("a%d += %s(ld(X + i + %d));" % (u, CV, u * W) for u in range(U))),
          "  %s s = 0; %s for (; i < n; i++) s += X[i];" % (sc, " ".join("s += %s(a%d);" % (HS, u) for u in range(U))),
          "  out[0] = (float)s;", "}"]
    return L


LOWER = {"matmul": lower_matmul, "softmax": lower_softmax, "layernorm": lower_layernorm, "reduce": lower_reduce}


def lower(ast_text):
    op, d = ast_to_sched(parse_sexpr(ast_text))
    return op, "\n".join(LOWER[op](d)) + "\n"


def main():
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("space")
    s = sp.add_parser("lower"); s.add_argument("--ast", required=True); s.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "space":
        print(json.dumps({op: {"knobs": SPACE[op], "start": sched_to_ast(op, START[op]), "signature": SIGS[op],
                               "semantics": SEMANTICS[op]} for op in SPACE}, ensure_ascii=False))
    else:
        try:
            op, src = lower(a.ast)
        except Exception as e:
            print(json.dumps({"ok": False, "error": "%s: %s" % (type(e).__name__, e)})); sys.exit(2)
        open(a.out, "w").write(src); print(json.dumps({"ok": True, "op": op}))


if __name__ == "__main__":
    main()
