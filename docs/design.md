# 设计：用 Aura 做“可验证的算子优化 agent”（CPU 替代实验）

## 0. 背景
无问芯穹（Infinigence）在 NVIDIA SOL-ExecBench（235 题，B200，分数 S=(Tb−Tsol)/((Tk−Tsol)+(Tb−Tsol))，基线 0.5、触顶 SOL 为 1）登顶。
其做法的要点：workspace / debug / eval 三容器隔离；技能库 + 共享记忆；Triton/PTX 多路径；
官方评测加固（清 L2、独立子进程、静态反作弊——14.5% 的 agent 提交被判作弊）；GPU 争用（排队中位数 24–40 分钟）。

问题：**Aura 能否做得更好？** 我们把用户给出的四个切入点落成一个可跑的小实验。

| 切入点 | 本仓库里的实现 |
|---|---|
| 1. 内核作为 Aura EDSL，AST 变异 + 增量编译 | 算法固定、**调度**是 S 表达式，常驻 Aura live AST（`kernel-cur`），变异走 `set-code`/`mutate:rebind`，`ast:snapshot`/`ast:restore` 回滚 |
| 2. 评测作为受控 fiber 操作 + 契约 | `eval-batch!` 每个候选一个 fiber 起隔离子进程；`kernel-legal?` 契约在评测前拦截非法 AST（不耗预算）；keval 做反作弊与数值门禁 |
| 3. 可演化的工作流 | 未做（时间所限），见“后续” |
| 4. 小算子上与“文本生成”对比 | matmul / softmax / layernorm / reduce × 3 种方法 × 3 种子，同一评测器、同一预算 |

## 1. 为什么是 CPU 替代
本机（box）**没有 GPU**（无 `nvidia-smi`），不可能跑 SOL-ExecBench。改为单核 AVX-512（Xeon，8 核，L2 16 MiB）。
**这不是 SOL-ExecBench，数字不可与其比较**，只用来比较“搜索方法”本身。

- 固定编译选项：`-std=gnu11 -O2 -march=native -fno-tree-vectorize -fno-unroll-loops -fno-strict-aliasing`
  ——关掉自动向量化/展开，向量化只能来自候选自己（Aura 侧来自 lowering 的 GCC 向量扩展，文本侧来自 intrinsics）。
- 计时串行化：`flock /tmp/akb-time.lock` + `taskset -c 7`，相当于文章里的“GPU 租约”，但公平排队，不会被占着不放。
- **T_b（基线）**：朴素源码 `-O3 -march=native -ffast-math`（编译器能自动向量化时的水平）。
- **T_SOL**：单核 roofline，`max(flops/峰值, 字节/带宽)`，峰值（AVX-512 FMA 微基准，~250 GFLOP/s）与按工作集测的读带宽见 `results/calib.json`。

## 2. 评测器 keval（三种方法共用）
1. 静态检查：include 白名单；禁止 asm、非 inline 的 static/全局状态、构造/析构、`optimize`/`target` 属性、I/O/计时/进程调用、引用 harness 符号、窥栈。
2. 符号检查（nm）：只允许导出 `kernel`；禁止任何数据/BSS/common 段符号（防缓存结果）；未定义符号白名单（libm + libmvec）。
3. 隔离子进程（rlimit CPU 120 s / 内存 2 GB）。harness 每个形状两套交替的随机输入、调用前用 NaN 填输出、调用后 memcmp 检查输入未被改、在 kernel 之后再算 double 参考；带“大偏移行”考验数值稳定性。
4. 计时：中位数，至少 N 次或 0.15 s；计时后再校验一次最后的输出。

## 3. 三种方法（预算 = 16 次评测，宽度 2）
- **aura-random**：Aura 结构化随机变异（order 交换相邻循环；其它旋钮 70% ±1 步、30% 随机跳）。
- **aura-llm**：LLM 只**提议旋钮改动**（JSON），Aura 负责契约、去重、评测、确认、回滚；LLM 输出不合法时随机补位。
- **text-llm**：LLM 直接写完整 C（可用 intrinsics），看得到历史与当前最优源码——即文章式的“文本生成”基线。
- Aura 侧接受规则：候选比 incumbent 快 ≥2% → 快照 → rebind → 用新输入复测候选与旧 incumbent，确认 <0.99× 才保留，否则 `ast:restore`。
  非法 / 重复的提议不消耗预算；连续 5 轮无新候选则停止。
- LLM：DeepSeek `deepseek-flash`（OpenAI 兼容），**关闭思考模式**——开启时 16k token 全部花在 reasoning 上，大部分回合拿不到代码。

## 4. 调度空间（tools/kgen.py）
- matmul：order(6 种) × tile-i/j/k × unroll × vec × micro（寄存器分块 RB 行 × vec·unroll 列，累加器常驻寄存器）
- softmax：passes(3/2-online) × exp(libm/多项式) × unroll × vec
- layernorm：stats(two-pass/sumsq) × acc(f32/f64) × unroll × vec
- reduce：acc(f32/f64) × unroll × vec

## 5. 后续
- 工作流演化：把变异策略本身放进 live AST，由同一门禁评估“策略改动”。
- 把 LLM 文本侧发现的技巧（scalef 指数、掩码尾、多累加器）“晋升”为新的 AST 构造/旋钮——这是 Aura 结合 LLM 最有希望的点。
- 真 GPU：lowering 到 Triton/CUDA，评测器换成 SOL-ExecBench 的 harness。
