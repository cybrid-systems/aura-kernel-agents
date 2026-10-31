# 本项目中发现的 Aura 问题与差距

> 当前实验 Aura pin：`1df09aa16`（aura main，含下述修复），运行于 `ghcr.io/cybrid-systems/dev:v1.0.9`；首轮实验用的是 `5af7fd023`（结果在 `results/old-pin-5af7fd023/`）。
> 2026-10-31 已在 cybrid-systems/aura 提 issue 并修复（见各条“状态”；aura 提交 `9318f772e`..`1df09aa16`）。升级 pin 后本仓库已去掉对应 workaround 并重跑全量实验，性能差异在噪声内（见 README“升级前后对比”）；回归测试 `tests/test_kernel_ast.aura`。

## A. 缺陷（bug）

1. **`string->symbol` 返回的是字符串**：`(symbol? (string->symbol "x"))` → `#f`，`(string? (string->symbol "x"))` → `#t`。
   后果：从 JSON/文本读进来的旋钮名转成 "symbol" 后与 `'tile-i` 这类字面量 `eq?`/`assq` 永远不等。
   旧 pin 上本项目在 `aura/kernel.aura` 里一律用 `(format "~a" x)` 比较名字绕过；升级后已改为 `string->symbol` + `assq`/`eq?`。
   这直接导致过一次 **搜索死循环**（所有候选都被契约判为“未知旋钮”且不耗预算），因此增加了“连续 5 轮无新候选则停止”的保护。
   **状态：已修复** —— [aura#4798](https://github.com/cybrid-systems/aura/issues/4798)：`string->symbol` 经 `read-from-string` intern，结果与 `'sym` `eq?`；非 symbol 文本（`"123"`、`"a b"`、`""`）仍回退为字符串。
2. 沿用 aura-eda-agents 已知问题：fiber 内需要重新 `let` 绑定外层变量（aura#4781），`json-parse` 只能在主 fiber 调（aura#4788）。
   **状态：两者在旧 pin `5af7fd023` 里就已修复**（aura#4781/#4788 已关闭）；重跑时已删除重新 `let` 绑定，`eval-batch!` 改为子 fiber 内直接读结果 JSON。

## B. 语言/库易用性

3. 没有有理数字面量：`1/3` 是解析错误（需写 `(/ 1 3)` 或小数）。
   **状态：诊断已修复，语言策略不变** —— [aura#4799](https://github.com/cybrid-systems/aura/issues/4799)：Aura 有意不支持有理数（整数 `/` 截断，见 aura#4499）；修的是嵌套在列表里时（`(list 1/3)`）诊断被吞成 `unexpected ')'` 的问题，现在会在字面量位置报 “rational literal '1/3' is not supported”。
4. 通道 API 名是 `channel:create`（不是常见的 `make-channel`/`channel:make`），文档里不易查到。
   **状态：已修复（提示）** —— [aura#4801](https://github.com/cybrid-systems/aura/issues/4801)：`make-channel`/`channel-send`/`channel-recv`/`channel-close` 等常见拼写的 unbound 报错会提示正确的 `channel:*` 名字（不新增别名 primitive）。
5. `(hash "k" v ...)` 带初值构造不可靠/不一致，旧代码统一用 `(hash)` + `hash-set!`，升级后改回 `(hash "status" st)`。
   **更正 + 状态：已修复** —— 复查发现偶数参数时构造是正确的；真正的问题是**奇数个参数时静默丢掉最后一个 key**（漏写一个值不报错）。[aura#4800](https://github.com/cybrid-systems/aura/issues/4800)：现在奇数参数直接报错。

## C. 与“像 Infinigence 那样做算子优化”相比的结构性差距（最重要）

6. **Aura 没有面向 C/PTX/Triton 的 lowering 后端**。内核在 Aura 里只能表示为“调度 AST”（数据），真正生成 C 的 `tools/kgen.py` 是 Python 写的。
   实验里最大的单因素是 lowering 质量：同一个调度，`memcpy` 式向量读写被 GCC（-fno-tree-vectorize）展开成逐 8 字节拷贝循环，比直接解引用慢约 5.6×；
   改完后 softmax 仍比 LLM 手写 intrinsics（roundscale+scalef 的 exp）慢 ~1.7×。**Aura 侧搜索的上限 = 人写的 lowering 的上限**。
   **状态：部分（第一步）** —— [aura#4802](https://github.com/cybrid-systems/aura/issues/4802)：新增 `std/cgen`，C 子集 EDSL（类型化函数、for/if/decl/set/op=、中缀、下标、cast、三元、f32），kernel 可以作为 Aura S-expression 直接 lower 成 C。PTX/Triton、性能注解、与 workspace rewrite 打通仍待做。
7. **增量编译只作用于 Aura 自身的代码**：`set-code`/`mutate:rebind` 让“改一个调度旋钮”在 Aura 里是 O(1) 的 AST 替换，但下游 C 仍要整文件 gcc 重编（~0.1–0.3 s/候选），GPU 场景里 nvcc/ptxas 的成本同样不受 Aura 增量编译影响。
   **状态：部分（第一步）** —— [aura#4803](https://github.com/cybrid-systems/aura/issues/4803)：`std/cgen` 的 `cgen-digest` + `cgen-build-cached!` 按生成文本内容摘要缓存构建结果，未变化的候选不再重编（进程内）。落盘缓存、编译选项入 key、按 AST 依赖只重建受影响函数仍待做。
8. **契约只能表达“结构合法”，表达不了“数值合法”**：例如 `reduce (acc f32)` 结构合法但数值不过关，只能靠外部 keval 的数值门禁在评测后才拒绝（消耗预算）。
9. 搜索空间是人定的：LLM 文本生成能用到空间之外的技巧（`_mm512_scalef_ps`、掩码尾处理、多累加器拆依赖链），Aura 侧搜索做不到，除非把这些技巧“晋升”为新的旋钮/AST 构造——这一步目前仍需人或 LLM 改 Python lowering，Aura 无法自举。
10. 没有原生的“受控资源”原语：计时公平性靠 `flock` + `taskset` 外部实现；文章里 GPU 租约/排队的问题在 Aura 里同样需要自己造。

## D. 重跑时的新发现（未提 issue）

11. `read-file` 对不存在的文件返回 `'()`，而 `'()` 在 Aura 里为真：`(if (safe (lambda () (read-file p)) #f) …)` 这种“文件存在吗”的写法永远成立，
    导致本仓库 `search-main.aura` 没写 `aura_runs.tsv` 表头（已改为 `string?` 判断）。新旧 pin 行为相同，且 `std/fs` 的 `file-exists?` 正是依赖这一约定，
    属既有设计而非回归，故未提 issue；`file-exists?` 需 `(require "std/fs")`，未 require 时只报 unbound，没有提示。
