#!/bin/bash
# 完整实验（nohup 跑）：编译器基线 + 标定 → Aura 两种方法（顺序）与文本基线（4 个算子并行）→ 复测 → 汇总
# 用法：LLM_ALLOW_NETWORK=1 nohup scripts/run_experiment.sh > runs/experiment.log 2>&1 &
set -u; cd "$(dirname "$0")/.."
SEEDS=${AKB_SEEDS:-1,2,3}; BUDGET=${AKB_BUDGET:-16}; OPS=${AKB_OPS:-matmul,softmax,layernorm,reduce}; PROV=${LLM_PROVIDER:-deepseek}
export AKB_SEEDS=$SEEDS AKB_BUDGET=$BUDGET AKB_OPS=$OPS LLM_PROVIDER=$PROV
echo "start $(date '+%F %T %Z') seeds=$SEEDS budget=$BUDGET ops=$OPS provider=$PROV"
scripts/dev.sh 'python3 tools/calib.py --out results/calib.json && python3 -c "
import json,sys; sys.path.insert(0,\"tools\"); import keval
out={}
for op in \"'$OPS'\".split(\",\"):
    xs=[keval.evaluate(op, keval.NAIVE[op], cflags=\"-std=gnu11 -O3 -march=native -ffast-math\".split()) for _ in range(3)]
    ms=sorted(x[\"ms_geo\"] for x in xs if x[\"status\"]==\"ok\"); out[op]={\"ms_geo\": ms[len(ms)//2], \"samples\": ms, \"cflags\": \"-O3 -march=native -ffast-math\"}
    print(op, out[op])
json.dump(out, open(\"results/compiler_baseline.json\",\"w\"), indent=1)"'
# 文本基线：每个算子一个进程并行（瓶颈是 LLM 延迟；计时段由 flock 串行）
for op in ${OPS//,/ }; do
  ( for s in ${SEEDS//,/ }; do scripts/dev.sh "python3 tools/text_baseline.py --op $op --seed $s --budget $BUDGET --dir runs/$op/text-llm/s$s --out runs/$op/text-llm/s$s/result.json"; done ) > runs/text-$op.log 2>&1 &
done
AKB_METHODS=aura-random,aura-llm AKB_OUT=results/aura_runs.tsv scripts/dev.sh 'scripts/aura.sh aura/search-main.aura'
wait
echo "search done $(date '+%F %T %Z')"
scripts/dev.sh 'python3 tools/summarize.py remeasure --runs runs --out results/final.tsv'
python3 tools/summarize.py report --runs runs --final results/final.tsv --calib results/calib.json --base results/compiler_baseline.json --out results/summary.md
echo "EXPERIMENT-DONE $(date '+%F %T %Z')"
