#!/usr/bin/env bash
# Diagnostics script for inference service: GPU + latency + memory sampling
# Usage: ./scripts/run_inference_diagnostics.sh [ITERATIONS] [HOST] [PORT]

set -euo pipefail

# Find a Python executable (prefer python3). Verify it runs code and has numpy.
PYTHON_CMD=""
PYTHON_ARGS="-"
for cmd in python3 python py; do
  if command -v "$cmd" >/dev/null 2>&1; then
    # verify it can execute a tiny statement
    if [ "$cmd" = "py" ]; then
      if $cmd -3 -c "import sys; import numpy" >/dev/null 2>&1; then
        PYTHON_CMD=$cmd
        PYTHON_ARGS='-3 -'
        break
      fi
    else
      if $cmd -c "import sys; import numpy" >/dev/null 2>&1; then
        PYTHON_CMD=$cmd
        PYTHON_ARGS='-'
        break
      fi
    fi
  fi
done
if [ -z "$PYTHON_CMD" ]; then
  echo "Warning: usable python with numpy not found in PATH. Using fallback (zero-filled PSF)." >&2
  PYTHON_CMD=""
fi

ITERATIONS=${1:-50}
HOST=${2:-localhost}
PORT=${3:-5001}
BASE_URL="http://${HOST}:${PORT}"

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="$ROOT_DIR/logs/inference_diagnostics"
PSF_FILE="$ROOT_DIR/simulador/sample_psf.bin"
# If using Windows python (py launcher), convert POSIX path like /c/... to C:/...
PSF_FILE_FOR_PY="$PSF_FILE"
if [ "$PYTHON_CMD" = "py" ]; then
  drive=$(echo "$PSF_FILE" | sed -E 's#^/([a-zA-Z])/.*#\1#')
  pathpart=$(echo "$PSF_FILE" | sed -E 's#^/[a-zA-Z]/(.*)#\1#')
  drive=$(echo "$drive" | tr '[:lower:]' '[:upper:]')
  PSF_FILE_FOR_PY="${drive}:/${pathpart}"
fi
RESP_DIR="$LOG_DIR/responses"

mkdir -p "$LOG_DIR" "$RESP_DIR"

echo "Logs: $LOG_DIR"

generate_psf(){
  echo "Generating sample PSF -> $PSF_FILE"
  # Try to use a POSIX python (not the Windows py launcher) if available; otherwise fall back to a zero-filled binary.
  # Use POSIX python only if available and path is not MSYS (/c/ or /mnt/c/).
  if [ -n "$PYTHON_CMD" ] && ! echo "$PSF_FILE" | grep -Eq '^/(c|mnt/c)/'; then
    $PYTHON_CMD $PYTHON_ARGS <<PY
import numpy as np
arr = np.random.rand(2,96,96).astype('float32')
open(r'$PSF_FILE','wb').write(arr.tobytes())
print('wrote', r'$PSF_FILE', len(arr.tobytes()))
PY
  else
    # create zeroed float32 array (2*96*96 elements * 4 bytes)
    SIZE=$((2*96*96*4))
    mkdir -p "$(dirname "$PSF_FILE")"
    dd if=/dev/zero of="$PSF_FILE" bs=$SIZE count=1 2>/dev/null || \
      (head -c $SIZE /dev/zero > "$PSF_FILE")
    echo "wrote $PSF_FILE (zero-filled) size=$SIZE"
  fi
}

start_monitors(){
  echo "Starting monitors (nvidia-smi and docker stats)"
  # nvidia-smi sampler
  (nvidia-smi -l 1 > "$LOG_DIR/nvidia_smi.log" 2>&1) &
  echo $! > "$LOG_DIR/nvidia_smi.pid"

  # docker stats sampler (1s interval)
  (
    while true; do
      date --iso-8601=seconds >> "$LOG_DIR/docker_stats.log" 2>&1 || true
      docker stats --no-stream --format "{{.Container}} {{.Name}} {{.CPUPerc}} {{.MemUsage}} {{.NetIO}} {{.BlockIO}}" >> "$LOG_DIR/docker_stats.log" 2>&1 || true
      sleep 1
    done
  ) &
  echo $! > "$LOG_DIR/docker_stats.pid"

}

stop_monitors(){
  echo "Stopping monitors"
  if [ -f "$LOG_DIR/nvidia_smi.pid" ]; then
    kill "$(cat $LOG_DIR/nvidia_smi.pid)" 2>/dev/null || true
    rm -f "$LOG_DIR/nvidia_smi.pid"
  fi
  if [ -f "$LOG_DIR/docker_stats.pid" ]; then
    kill "$(cat $LOG_DIR/docker_stats.pid)" 2>/dev/null || true
    rm -f "$LOG_DIR/docker_stats.pid"
  fi
}

run_stress(){
  MODEL=$1
  TIMES_FILE="$LOG_DIR/${MODEL}_times.txt"
  STATUS_FILE="$LOG_DIR/${MODEL}_status.txt"
  > "$TIMES_FILE"
  > "$STATUS_FILE"

  echo "Running stress test for model=$MODEL iterations=$ITERATIONS"
  for i in $(seq 1 $ITERATIONS); do
    start=$(date +%s.%N)
    # Save first 3 responses for inspection
    if [ "$i" -le 3 ]; then
      OUT_FILE="$RESP_DIR/${MODEL}_resp_${i}.json"
      status=$(curl -s -w "%{http_code}" -o "$OUT_FILE" -X POST "$BASE_URL/predict?model=$MODEL" --data-binary @$PSF_FILE -H "Content-Type: application/octet-stream" )
    else
      status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/predict?model=$MODEL" --data-binary @$PSF_FILE -H "Content-Type: application/octet-stream" )
    fi
    end=$(date +%s.%N)
    # compute elapsed using python for float accuracy
    if [ -n "$PYTHON_CMD" ] && ! echo "$LOG_DIR" | grep -Eq '^/(c|mnt/c)/'; then
      elapsed=$($PYTHON_CMD $PYTHON_ARGS <<PY
start=float('$start')
end=float('$end')
print(end-start)
PY
)
    else
      elapsed=$(awk "BEGIN{print $end - $start}")
    fi
    echo "$elapsed" >> "$TIMES_FILE"
    echo "$status" >> "$STATUS_FILE"
    printf "%s iter %d: status=%s time=%s\n" "$(date --iso-8601=seconds)" "$i" "$status" "$elapsed"
  done
}

compute_stats(){
  FILE=$1
  # Use Python only if available and the path is not MSYS (/c/ or /mnt/c/). Otherwise use awk fallback.
  if [ -n "$PYTHON_CMD" ] && ! echo "$FILE" | grep -Eq '^/(c|mnt/c)/'; then
$PYTHON_CMD $PYTHON_ARGS <<PY
import sys,statistics
data=open('$FILE').read().strip().split() or []
data=[float(x) for x in data]
if not data:
    print('no data')
    sys.exit(0)
data.sort()
import math
def pct(p):
    k=int(math.ceil(p/100.0*len(data)))-1
    k=max(0,min(k,len(data)-1))
    return data[k]
print('count',len(data))
print('mean',statistics.mean(data))
print('median',statistics.median(data))
print('p90',pct(90))
print('p95',pct(95))
print('min',data[0])
print('max',data[-1])
PY
  else
    # fallback to awk: compute count, mean, min, max, median approx, p90 approx
    if [ ! -f "$FILE" ]; then
      echo "no data"
      return
    fi
    sort -n "$FILE" > "$FILE.sorted"
    N=$(wc -l < "$FILE.sorted" | tr -d ' ')
    if [ "$N" -eq 0 ]; then
      echo "no data"
      return
    fi
    sum=$(awk '{s+=$1}END{printf "%f",s}' "$FILE.sorted")
    mean=$(awk -v s="$sum" -v n="$N" 'BEGIN{printf "%f", s/n}')
    min=$(awk 'NR==1{print $1; exit}' "$FILE.sorted")
    max=$(awk 'END{print $1}' "$FILE.sorted")
    # median
    mid=$(( (N+1)/2 ))
    median=$(awk -v m=$mid 'NR==m{print $1; exit}' "$FILE.sorted")
    # p90 index
    p90idx=$(( (90*N+99)/100 ))
    p90=$(awk -v p=$p90idx 'NR==p{print $1; exit}' "$FILE.sorted")
    echo "count $N"
    echo "mean $mean"
    echo "median $median"
    echo "p90 $p90"
    echo "min $min"
    echo "max $max"
  fi
}

trap 'echo "Interrupted, stopping monitors"; stop_monitors; exit 1' INT TERM

generate_psf
start_monitors

run_stress phase_diversity
run_stress resnet10

echo "Stopping monitors and gathering summaries"
stop_monitors

echo "Phase_diversity stats:"; compute_stats "$LOG_DIR/phase_diversity_times.txt"
echo "Resnet10 stats:"; compute_stats "$LOG_DIR/resnet10_times.txt"

echo "Saved responses in: $RESP_DIR"
echo "Saved logs in: $LOG_DIR"

echo "Done."
