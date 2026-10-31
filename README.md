# aura-kernel-agents

验证问题：**无问芯穹在 NVIDIA SOL-ExecBench 登顶的“agent 写算子”流程，用 Aura 能不能做得更好？**
（文章：<https://mp.weixin.qq.com/s/hPrW6K4OLS0sAyAWLvCmWg>）

> ⚠️ **CPU 替代，非 SOL-ExecBench。** 运行环境没有 GPU，本仓库在单核 AVX-512 CPU 上复刻“搜索 + 隔离评测 + 反作弊 + SOL 类比分数”的方法学，
> 只用来比较**搜索方法**，数字不能与 SOL-ExecBench 榜单比较。

## 做了什么
| 文章/切入点 | 本仓库 |
|---|---|
| 内核作为 Aura EDSL、AST 变异 | 调度是 S 表达式，常驻 Aura live AST；`set-code`/`mutate:rebind` 变异，`ast:snapshot`/`ast:restore` 回滚（`aura/kernel.aura`） |
| 评测隔离 + 反作弊 | `tools/keval.py`：静态规则 + nm 符号检查 + 子进程 rlimit + NaN 毒化输出/输入篡改检查/事后参考值；计时 `flock`+`taskset` 串行 |
| 契约 | `kernel-legal?` 在评测前拦截非法调度（不耗预算） |
| 文本生成基线 | `tools/text_baseline.py`：LLM 直接写 C（可用 intrinsics），看得到历史和当前最优 |
| SOL 分数 | `tools/calib.py` 单核 roofline 给 T_SOL；T_b = 朴素源码 `-O3 -march=native -ffast-math` |

方法（同一评测器、预算 16 次评测、3 个种子；LLM = DeepSeek `deepseek-flash`，关闭思考模式）：
- **aura-random**：Aura 结构化随机变异
- **aura-llm**：LLM 只提议旋钮改动，Aura 负责契约/去重/评测/复测确认/回滚
- **text-llm**：LLM 写完整 C

## 结果（2026-10-31 重跑，Aura pin `1df09aa16`，详见 `results/summary.md`）
最终内核用新随机输入复测 3 次取中位，3 个种子平均（ms，越小越好）。SOL 类比分：0.5 = 编译器基线，1 = 单核 roofline。
本次重跑把 Aura 升级到 main `1df09aa16`（含 aura#4798 `string->symbol`、aura#4800 `hash` 修复），并去掉了对应的 workaround（见下文“升级前后对比”）；
旧 pin `5af7fd023` 的结果原样保存在 `results/old-pin-5af7fd023/`。

| 算子 | 起点 | 编译器基线 T_b | T_SOL | aura-random | aura-llm | text-llm |
|---|---|---|---|---|---|---|
| matmul 256³ 等 | 6.47 | 5.49 | 0.096 | 1.124（分 0.84） | 0.243（0.97） | **0.120（1.00）** |
| softmax | 2.15 | 0.196 | 0.048 | 0.182（0.52） | 0.164（0.56） | **0.112（0.70）** |
| layernorm | 2.57 | 0.090 | 0.052 | 0.088（0.51） | 0.084（0.54） | 0.087（0.52） |
| reduce（带宽受限） | 19.37* | 0.415 | 0.361 | 0.360 | 0.371 | 0.353 |

\* reduce 起点（朴素 -O2，32 MB 流式读）本次测得 19.4 ms，旧实验 9.7 ms：同一份源码，差异来自与 ORFS 实验争抢内存带宽（重跑期间 load 8–9 / 8 核）。

| 每次运行平均成本 | aura-random | aura-llm | text-llm |
|---|---|---|---|
| 墙钟 | 13–30 s | 28–64 s | 61–169 s |
| LLM token | 0 | 5.7k–8.9k | 32k–110k |
| 无效评测（编译错/数值错/作弊）占 48 次 | 0–11 | 0–12 | 最多 28（layernorm 27 次数值错 + 1 次编译错） |

### 升级前后对比（pin `5af7fd023` → `1df09aa16`）
去掉的 workaround（都已由回归测试 `tests/test_kernel_ast.aura` 覆盖；该测试在旧 pin 上 6 项失败、新 pin 全过）：
- 旋钮名比较：`(format "~a" …)` 字符串比较 → `string->symbol` + `assq`/`eq?`（aura#4798）；
- `(hash)` + `hash-set!` → `(hash "status" st)`（aura#4800：奇数参数现在报错，不再静默丢 key）；
- 子 fiber 只回传目录、主 fiber 统一 `json-parse` → 子 fiber 内直接读结果 JSON（aura#4788，旧 pin 已含修复）；
- fiber 体内重新 `let` 绑定外层参数 → 删除（aura#4781，旧 pin 已含修复）。

最终内核性能（ms，3 种子平均）。“同场复测”= 重跑结束后在同一时段、同一核上把新旧两批最优内核各复测一次（`results/same-session-remeasure/`）：

| 算子 / 方法 | 旧（原始） | 新（原始） | 旧（同场复测） | 新（同场复测） | 新/旧（同场） |
|---|---|---|---|---|---|
| matmul aura-random | 1.133 | 1.124 | 1.236 | 1.122 | 0.91 |
| matmul aura-llm | 0.230 | 0.243 | 0.240 | 0.246 | 1.03 |
| matmul text-llm | 0.125 | 0.120 | 0.134 | 0.120 | 0.90 |
| softmax aura-random | 0.159 | 0.182 | 0.165 | 0.166 | 1.01 |
| softmax aura-llm | 0.158 | 0.164 | 0.160 | 0.164 | 1.02 |
| softmax text-llm | 0.113 | 0.112 | 0.124 | 0.106 | 0.86 |
| layernorm aura-random | 0.093 | 0.088 | 0.092 | 0.101 | 1.10 |
| layernorm aura-llm | 0.095 | 0.084 | 0.089 | 0.102 | 1.15 |
| layernorm text-llm | 0.079 | 0.087 | 0.083 | 0.095 | 1.15 |
| reduce aura-random | 0.364 | 0.360 | 0.379 | 0.356 | 0.94 |
| reduce aura-llm | 0.413 | 0.371 | 0.360 | 0.364 | 1.01 |
| reduce text-llm | 0.352 | 0.353 | 0.339 | 0.331 | 0.98 |

结论：**升级与去 workaround 不改变搜索行为，性能差异全在噪声内。**
- 噪声下限：有 7 组运行新旧最优内核**完全相同**（matmul aura-random 3 个种子全同，aura-random 是种子确定的），同一内核同场复测比值仍在 0.81–1.15 之间；
  text-llm 根本不经过 Aura，其新旧比值也在 0.86–1.15。所以表中任何 ±15% 以内的差异都不能归因于 Aura 升级。
- 搜索计数几乎不变：契约拦截 + 去重（不耗预算）旧 14 + 45 = 59，新 16 + 44 = 60；KEEP/ROLLBACK 旧 78/9，新 79/9；`aura-error`/`lower-error` 两次都是 0。
- 唯一的排名变化：layernorm 上 aura-llm（0.084）略好于 text-llm（0.087），但同场复测里两者都是 ~0.095–0.102，属噪声，不作为结论。
- `std/cgen`（aura#4802 第一步）这次**没有**接入：lowering 仍是 `tools/kgen.py`。把它换成 cgen 会同时改变生成的 C，就无法做上面这种“只换 Aura 版本”的对照；留作下一步单独实验。
- 重跑中发现本仓库自己的一个 bug：`read-file` 对不存在的文件返回 `'()`（Aura 里 `'()` 为真），`search-main.aura` 因而没写 `aura_runs.tsv` 表头；
  这是 Aura 的既有约定（`std/fs` 的 `file-exists?` 也基于它，新旧 pin 行为相同），不是回归，已改为 `string?` 判断并补回表头。

噪声说明：重跑期间 ORFS 实验（openroad 多线程，~350% CPU）与其他构建并行，load 8–9 / 8 核。计时段由 `flock` 串行并 `taskset -c 7` 绑核，
但无法独占核心与内存带宽；同一内核复测波动约 ±15%（reduce 起点甚至 2×）。

### 诚实的结论
1. **最终性能：文本生成赢**（matmul/softmax 领先 Aura 侧 1.5–2×；layernorm、reduce 在噪声内持平，旧 pin 时 layernorm 文本侧领先 1.2×）。原因不是搜索，是**表达能力**：
   LLM 会用调度空间之外的技巧（`_mm512_scalef_ps` 指数、掩码尾、多累加器拆依赖链、寄存器分块微内核）；Aura 侧的上限 = 人写的 lowering 的上限。
2. **成本与可靠性：Aura 侧赢**。token 少一个数量级（aura-llm ~7k vs text ~60k），墙钟在 matmul/softmax/layernorm 上快 2.4–5.8×（reduce 上 aura-llm 64 s 与 text 61 s 持平，因 LLM 提议轮数多 + 带宽争抢）；契约在评测前拦掉了非法/重复提议（aura-llm 共 60 次，不耗预算；旧 pin 59 次），
   aura 的无效评测只来自数值门禁（f32 累加）；文本侧在 layernorm 上 48 次里 27 次数值错误（旧 pin 21 次）。
3. **LLM 提议 + Aura 契约 > 纯随机**：matmul 上 aura-llm 0.24 ms vs aura-random 1.12 ms（随机游走在“vec 需要 j 最内层”的契约壁垒前卡住，种子 3 停在 ~2.8 ms，新旧 pin 一致）。小空间（reduce/layernorm/softmax）两者相当。
4. lowering 质量是单一最大因素：一次修复（向量读写从 `memcpy` 改为直接解引用）让 softmax 同一调度快 5.6×。
5. 值得做的结合方向：把文本侧发现的技巧**晋升**为 Aura AST 构造/旋钮（文本 LLM 负责探索、Aura 负责可验证的复用），以及让 Aura 有自己的 lowering 后端。


## Aura 的问题与差距
见 [`docs/aura-issues.md`](docs/aura-issues.md)（已提 aura#4798–#4803 并修复/部分实现，本次重跑已用上修复；`string->symbol` 返回字符串导致的死循环、无有理数字面量、无 C/GPU lowering 后端、增量编译不覆盖下游 C、契约无法表达数值合法性……）。设计细节见 [`docs/design.md`](docs/design.md)。

## 复现
```bash
# 需要 docker 镜像 ghcr.io/cybrid-systems/dev:v1.0.9 与 aura 二进制（pin 1df09aa16；scripts/dev.sh 默认 AURA_PIN=/workspace/aura-kab-bin）
scripts/dev.sh 'scripts/aura.sh tests/test_kernel_ast.aura'  # 去掉 workaround 后的回归（需 aura ≥ 1df09aa16）
scripts/dev.sh 'python3 tests/test_keval.py'               # 评测器自测（反作弊/数值门禁）
DEEPSEEK_API_KEY=... LLM_ALLOW_NETWORK=1 nohup scripts/run_experiment.sh > runs/experiment.log 2>&1 &
```
目录：`aura/`（Aura 侧搜索）、`tools/`（kgen lowering、harness、keval、LLM、汇总）、`results/`（汇总、标定、每次运行的 trials / 最优内核 / LLM 调用日志，不含 key）。

许可证：Apache-2.0
