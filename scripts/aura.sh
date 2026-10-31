#!/usr/bin/env bash
# 把 aura/common.aura + 模块 + 入口拼成一个文件后用 Aura 运行（与 aura-loading-prototype 相同的做法）。
#   AURA_BIN=/path/to/aura scripts/aura.sh aura/pd-main.aura
# 入口文件第一行可以写 ";; uses: pd.aura eco.aura" 声明依赖的模块。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
SRC="${1:?usage: scripts/aura.sh aura/<entry>.aura}"; mkdir -p build
FILE="build/$(basename "$SRC" .aura).all.aura"
USES=$(sed -n '1s/^;; uses://p' "$SRC")
{ echo '(require "std/process" all:)'; cat aura/common.aura; for m in $USES; do cat "aura/$m"; done; cat "$SRC"; } > "$FILE"
: "${AURA_BIN:?set AURA_BIN to an Aura binary}"
if [ -z "${AURA_PATH:-}" ] && [ -n "${AURA_RUNTIME_DIR:-}" ]; then export AURA_PATH="$AURA_RUNTIME_DIR/lib"; fi
exec env AURA_SANDBOX=off AURA_PIPELINE_STRICT=0 timeout "${AURA_TIMEOUT:-3600}" "$AURA_BIN" --load "$FILE"
