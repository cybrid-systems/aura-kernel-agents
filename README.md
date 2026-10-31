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

## 结果（2026-10-31，详见 `results/summary.md`）
最终内核用新随机输入复测 3 次取中位，3 个种子平均（ms，越小越好）。SOL 类比分：0.5 = 编译器基线，1 = 单核 roofline。

| 算子 | 起点 | 编译器基线 T_b | T_SOL | aura-random | aura-llm | text-llm |
|---|---|---|---|---|---|---|
| matmul 256³ 等 | 7.45 | 5.75 | 0.096 | 1.133（分 0.84） | 0.230（0.98） | **0.125（0.99）** |
| softmax | 2.17 | 0.211 | 0.049 | 0.159（0.59） | 0.158（0.60） | **0.113（0.72）** |
| layernorm | 2.65 | 0.097 | 0.052 | 0.093（0.52） | 0.095（0.51） | **0.079（0.62）** |
| reduce（带宽受限） | 9.67 | 0.432 | 0.331 | 0.364 | 0.413 | 0.352 |

| 每次运行平均成本 | aura-random | aura-llm | text-llm |
|---|---|---|---|
| 墙钟 | 14–22 s | 26–38 s | 65–135 s |
| LLM token | 0 | 5.8k–8.2k | 35k–85k |
| 无效评测（编译错/数值错/作弊）占 48 次 | 0–10 | 0–8 | 最多 23（layernorm 21 次数值错；matmul 1 次 `#pragma GCC optimize` 被判作弊） |

### 诚实的结论
1. **最终性能：文本生成赢**（matmul/softmax/layernorm 领先 Aura 侧 1.2–1.8×，reduce 在噪声内持平）。原因不是搜索，是**表达能力**：
   LLM 会用调度空间之外的技巧（`_mm512_scalef_ps` 指数、掩码尾、多累加器拆依赖链、寄存器分块微内核）；Aura 侧的上限 = 人写的 lowering 的上限。
2. **成本与可靠性：Aura 侧赢**。token 少一个数量级（aura-llm ~7k vs text ~60k），墙钟快 3–5×；契约在评测前拦掉了非法/重复提议（aura-llm 共 59 次，不耗预算），
   aura 的无效评测只来自数值门禁（f32 累加）；文本侧在 layernorm 上 48 次里 21 次数值错误。
3. **LLM 提议 + Aura 契约 > 纯随机**：matmul 上 aura-llm 0.23 ms vs aura-random 1.13 ms（随机游走在“vec 需要 j 最内层”的契约壁垒前卡住，种子 3 停在 2.77 ms）。小空间（reduce/layernorm/softmax）两者相当。
4. lowering 质量是单一最大因素：一次修复（向量读写从 `memcpy` 改为直接解引用）让 softmax 同一调度快 5.6×。
5. 值得做的结合方向：把文本侧发现的技巧**晋升**为 Aura AST 构造/旋钮（文本 LLM 负责探索、Aura 负责可验证的复用），以及让 Aura 有自己的 lowering 后端。

噪声说明：实验期间 box 上并行跑着另一个 ORFS 实验（load ~4）；计时已串行并绑核，但 reduce（32 MB 流式读，受共享 L3/内存带宽影响）复测波动约 ±15%，其差异不显著。

## Aura 的问题与差距
见 [`docs/aura-issues.md`](docs/aura-issues.md)（`string->symbol` 返回字符串导致的死循环、无有理数字面量、无 C/GPU lowering 后端、增量编译不覆盖下游 C、契约无法表达数值合法性……）。设计细节见 [`docs/design.md`](docs/design.md)。

## 复现
```bash
# 需要 docker 镜像 ghcr.io/cybrid-systems/dev:v1.0.9 与 aura 二进制（AURA_BIN，pin 5af7fd023）
scripts/dev.sh 'python3 tests/test_keval.py'               # 评测器自测（反作弊/数值门禁）
DEEPSEEK_API_KEY=... LLM_ALLOW_NETWORK=1 nohup scripts/run_experiment.sh > runs/experiment.log 2>&1 &
```
目录：`aura/`（Aura 侧搜索）、`tools/`（kgen lowering、harness、keval、LLM、汇总）、`results/`（汇总、标定、每次运行的 trials / 最优内核 / LLM 调用日志，不含 key）。

许可证：Apache-2.0
