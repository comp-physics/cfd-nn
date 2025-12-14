#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SSH_CONNECTION:-}" && -z "${SSH_TTY:-}" ]]; then
  echo "This script is intended to be run inside an existing SSH session on the cluster login node."
  exit 2
fi

JOBID="${1:-}"
if [[ -z "${JOBID}" ]]; then
  JOBID="$(sacct -X -n -u "$USER" --format=JobID --state=R,CG,CD,F,TO,PR,CA 2>/dev/null \
            | awk 'NF{print $1}' | head -n 1 || true)"
  [[ -z "${JOBID}" ]] && JOBID="$(squeue -u "$USER" -h -o "%i" | head -n 1 || true)"
  [[ -z "${JOBID}" ]] && { echo "usage: slurm_packet.sh JOBID (no jobs found)"; exit 2; }
fi

echo "=== sacct ==="
sacct -j "$JOBID" --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocCPUS,Timelimit -P 2>/dev/null || true

# Try to locate stdout/stderr via scontrol (works best for running/recent jobs)
JOBINFO="$(scontrol show job "$JOBID" 2>/dev/null || true)"
OUT_FILE="$(printf '%s\n' "$JOBINFO" | tr ' ' '\n' | awk -F= '$1=="StdOut"{print $2}' | head -n1)"
ERR_FILE="$(printf '%s\n' "$JOBINFO" | tr ' ' '\n' | awk -F= '$1=="StdErr"{print $2}' | head -n1)"
CMD_LINE="$(printf '%s\n' "$JOBINFO" | tr ' ' '\n' | awk -F= '$1=="Command"{print $2}' | head -n1)"

# Fallback: if StdOut/StdErr not present (older configs), fall back to common log dir patterns
if [[ -z "${OUT_FILE:-}" ]]; then
  OUT_FILE="$(compgen -G "logs/*${JOBID}*.out" | head -n 1 || true)"
fi
if [[ -z "${ERR_FILE:-}" ]]; then
  ERR_FILE="$(compgen -G "logs/*${JOBID}*.err" | head -n 1 || true)"
fi

echo -e "\n=== job command (from scontrol, if available) ==="
echo "${CMD_LINE:-"(not available)"}"

echo -e "\n=== stderr tail (120) ==="
if [[ -n "${ERR_FILE:-}" && -r "${ERR_FILE:-/dev/null}" ]]; then
  echo "(file: $ERR_FILE)"
  tail -n 120 "$ERR_FILE"
else
  echo "(stderr file not found/readable)"
fi

echo -e "\n=== stdout tail (80) ==="
if [[ -n "${OUT_FILE:-}" && -r "${OUT_FILE:-/dev/null}" ]]; then
  echo "(file: $OUT_FILE)"
  tail -n 80 "$OUT_FILE"
else
  echo "(stdout file not found/readable)"
fi

echo -e "\n=== error-like matches (first 200 lines) ==="
PAT='(error|fatal|assert|nan|inf|segfault|SIG|Traceback|undefined reference|CMake Error|FAILED)'
if command -v rg >/dev/null 2>&1; then
  if [[ -n "${OUT_FILE:-}" && -r "${OUT_FILE:-/dev/null}" ]]; then
    rg -n --context 3 "$PAT" "$OUT_FILE" | head -n 200 || true
  fi
  if [[ -n "${ERR_FILE:-}" && -r "${ERR_FILE:-/dev/null}" ]]; then
    rg -n --context 3 "$PAT" "$ERR_FILE" | head -n 200 || true
  fi
else
  echo "(ripgrep 'rg' not found; skipping pattern scan)"
fi