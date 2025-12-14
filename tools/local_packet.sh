#!/usr/bin/env bash
set -euo pipefail

CMD="${*:?usage: local_packet.sh <command...>}"

echo "=== cmd ==="
echo "$CMD"

echo -e "\n=== env ==="
uname -a || true
command -v gcc >/dev/null 2>&1 && gcc --version | head -n 2 || true
command -v clang >/dev/null 2>&1 && clang --version | head -n 2 || true
command -v cmake >/dev/null 2>&1 && cmake --version | head -n 1 || true
command -v julia >/dev/null 2>&1 && julia --version || true

echo -e "\n=== run (tail) ==="
set +e
bash -lc "$CMD" 2>&1 | tee /tmp/local_run.log
EC=${PIPESTATUS[0]}
set -e
echo -e "\n(exit=$EC)"
tail -n 120 /tmp/local_run.log

echo -e "\n=== error-like matches (first 200) ==="
PAT='(error|fatal|assert|nan|inf|segfault|Traceback|FAILED|CMake Error|undefined reference|runtime error|sanitizer)'
if command -v rg >/dev/null 2>&1; then
  rg -n --context 3 "$PAT" /tmp/local_run.log | head -n 200 || true
else
  echo "(ripgrep 'rg' not found; skipping pattern scan)"
fi

exit "$EC"