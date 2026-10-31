# 本项目中发现的 Aura 问题与差距

> 只记录，未提 issue（按要求由维护者决定是否开）。Aura pin：`5af7fd023`（与 aura-eda-agents 相同），运行于 `ghcr.io/cybrid-systems/dev:v1.0.9`。

## A. 缺陷（bug）

1. **`string->symbol` 返回的是字符串**：`(symbol? (string->symbol "x"))` → `#f`，`(string? (string->symbol "x"))` → `#t`。
   后果：从 JSON/文本读进来的旋钮名转成 "symbol" 后与 `'tile-i` 这类字面量 `eq?`/`assq` 永远不等。
   本项目在 `aura/kernel.aura` 里一律用 `(format "~a" x)` 比较名字绕过。
   这直接导致过一次 **搜索死循环**（所有候选都被契约判为“未知旋钮”且不耗预算），因此增加了“连续 5 轮无新候选则停止”的保护。
2. 沿用 aura-eda-agents 已知问题：fiber 内需要重新 `let` 绑定外层变量（aura#4781），`json-parse` 只能在主 fiber 调（aura#4788）。
   本项目的 `eval-batch!` 由子 fiber 只回传目录，主 fiber 统一读 JSON。

## B. 语言/库易用性

3. 没有有理数字面量：`1/3` 是解析错误（需写 `(/ 1 3)` 或小数）。
4. 通道 API 名是 `channel:create`（不是常见的 `make-channel`/`channel:make`），文档里不易查到。
5. `(hash "k" v ...)` 带初值构造不可靠/不一致，本项目统一用 `(hash)` + `hash-set!`。

## C. 与“像 Infinigence 那样做算子优化”相比的结构性差距（最重要）

6. **Aura 没有面向 C/PTX/Triton 的 lowering 后端**。内核在 Aura 里只能表示为“调度 AST”（数据），真正生成 C 的 `tools/kgen.py` 是 Python 写的。
   实验里最大的单因素是 lowering 质量：同一个调度，`memcpy` 式向量读写被 GCC（-fno-tree-vectorize）展开成逐 8 字节拷贝循环，比直接解引用慢约 5.6×；
   改完后 softmax 仍比 LLM 手写 intrinsics（roundscale+scalef 的 exp）慢 ~1.7×。**Aura 侧搜索的上限 = 人写的 lowering 的上限**。
7. **增量编译只作用于 Aura 自身的代码**：`set-code`/`mutate:rebind` 让“改一个调度旋钮”在 Aura 里是 O(1) 的 AST 替换，但下游 C 仍要整文件 gcc 重编（~0.1–0.3 s/候选），GPU 场景里 nvcc/ptxas 的成本同样不受 Aura 增量编译影响。
8. **契约只能表达“结构合法”，表达不了“数值合法”**：例如 `reduce (acc f32)` 结构合法但数值不过关，只能靠外部 keval 的数值门禁在评测后才拒绝（消耗预算）。
9. 搜索空间是人定的：LLM 文本生成能用到空间之外的技巧（`_mm512_scalef_ps`、掩码尾处理、多累加器拆依赖链），Aura 侧搜索做不到，除非把这些技巧“晋升”为新的旋钮/AST 构造——这一步目前仍需人或 LLM 改 Python lowering，Aura 无法自举。
10. 没有原生的“受控资源”原语：计时公平性靠 `flock` + `taskset` 外部实现；文章里 GPU 租约/排队的问题在 Aura 里同样需要自己造。
