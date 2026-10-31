#!/bin/bash
# 在 Aura 开发镜像里运行命令（gcc 16 / python3 与 Aura 运行时一致）。
# --network host：只有 LLM 方法需要联网（默认 bridge 网络无 DNS）。key 只按变量名透传，不出现在命令行里。
B=${AURA_PIN:-/workspace/aura-eda-bin}
exec sudo -n docker run --rm -i --memory=6g --memory-swap=6g --network host --entrypoint bash --user 1000:1000 -e HOME=/tmp \
  -v /workspace:/workspace -v /tmp:/tmp -w "$PWD" -e AURA_SANDBOX=off -e AURA_PIPELINE_STRICT=0 \
  -e AURA_BIN=$B/aura -e AURA_LIB=$B/lib -e AURA_PATH=$B/lib \
  -e MINIMAX_API_KEY -e GLM_API_KEY -e DEEPSEEK_API_KEY -e LLM_ALLOW_NETWORK -e LLM_PROVIDER \
  -e AKB_CORE -e AKB_BUDGET -e AKB_SEEDS -e AKB_OPS -e AKB_METHODS -e AKB_WIDTH -e AKB_OUT \
  ghcr.io/cybrid-systems/dev:v1.0.9 -c "$*"
