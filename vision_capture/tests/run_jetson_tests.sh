#!/bin/bash
# run_jetson_tests.sh - full accuracy + efficiency test sequence on the Jetson.

TEST_DIR="$1"
ENGINE="$2"
WEIGHTS="${3:-controllers/pupil_lite_best.pth}"

if [ -z "$TEST_DIR" ] || [ -z "$ENGINE" ]; then
  echo "usage: bash tests/run_jetson_tests.sh TEST_DIR ENGINE [WEIGHTS]"
  exit 1
fi

OUT="results_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"

echo "== Fixing clocks for reproducible numbers =="
sudo nvpmodel -m 0
sudo jetson_clocks
nvpmodel -q > "$OUT/nvpmodel.txt" 2>&1

# Allows Python to import the controllers folder from the root directory
export PYTHONPATH="."

for B in torch32 torch16 trt; do
  echo
  echo "################ backend: $B ################"

  echo "-- accuracy on the test split"
  python3 tests/eval_accuracy.py --test-dir "$TEST_DIR" --backend "$B" \
      --engine "$ENGINE" --weights "$WEIGHTS" --out "$OUT/acc_$B.csv" || exit 1

  echo "-- efficiency (tegrastats logging in the background)"
  sudo pkill tegrastats 2>/dev/null || true
  sudo stdbuf -oL tegrastats --interval 500 > "$OUT/tegra_$B.log" &
  sleep 3
  python3 tests/bench_efficiency.py --backend "$B" --engine "$ENGINE" --weights "$WEIGHTS" \
      --images "$TEST_DIR/images" --n 1500 --two-eyes --out "$OUT/bench_$B.json" || exit 1
  sleep 2
  sudo pkill tegrastats
  python3 tests/tegra_summary.py "$OUT/tegra_$B.log" --trim 4 > "$OUT/tegra_$B.json"

  echo "-- cool-down 30 s"
  sleep 30
done

echo
echo "Done. Results in $OUT/"
echo "Copy it to your PC, then run analyze_results.py (see its header for the command)."