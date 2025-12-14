#!/usr/bin/env bash
set -euo pipefail

RUNID="${1:-}"
if [[ -z "$RUNID" ]]; then
  RUNID="$(gh run list --limit 20 --json databaseId,conclusion -q '.[] | select(.conclusion=="failure") | .databaseId' | head -n 1 || true)"
  [[ -z "$RUNID" ]] && { echo "usage: ci_packet.sh RUN_ID (no failed runs found)"; exit 2; }
fi

echo "=== run id ==="
echo "$RUNID"

echo -e "\n=== failed steps only (head 260) ==="
gh run view "$RUNID" --log-failed | sed -n '1,260p'

echo -e "\n=== error-like matches (first 200) ==="
PAT='(error:|fatal:|FAILED|AssertionError|undefined reference|CMake Error|ninja: build stopped|Traceback|sanitizer|runtime error)'
gh run view "$RUNID" --log-failed \
  | rg -n --context 3 "$PAT" \
  | head -n 200 || true