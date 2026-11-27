#!/usr/bin/env bash
# Benchmark lengkap #12 dalam satu perintah, untuk dijalankan di laptop uji (Wisam). Pemilik: Gavra.
#
#   bash bench/run_all.sh            # penuh
#   QUICK=1 bash bench/run_all.sh    # uji cepat citra kecil (hanya cek skrip jalan; JANGAN dipakai di laporan)
#   STRONG=1 bash bench/run_all.sh   # ikut ulang strong scaling utama (K=8, 4 ukuran)
#
# Jalankan dari akar repo, di WSL/Ubuntu, dengan charger terpasang dan aplikasi lain ditutup.
# Hasil: bench/results/<TAG>_*.csv, docs/BENCH_*.md, docs/BENCH_scaling_<TAG>.md, dan satu arsip
# bench_hasil_<TAG>.tar.gz yang tinggal dikirim ke Gavra. Tidak ada angka yang diisi skrip ini;
# langkah yang gagal dicatat di ringkasan akhir, bukan diam-diam dilewati.
#
# Variabel (opsional): THREADS (mis. 1,2,4,8,16; bawaan auto), SENTINEL (daftar PNG dipisah spasi,
# bawaan data/sentinel*.png), OUTDIR, DOCDIR, TAG.
set -u
cd "$(dirname "$0")/.." || exit 1

QUICK=${QUICK:-0}; STRONG=${STRONG:-0}
OUTDIR=${OUTDIR:-bench/results}; DOCDIR=${DOCDIR:-docs}
THREADS=${THREADS:-auto}
STAMP=$(date +%Y%m%d_%H%M)
TAG=${TAG:-run_${STAMP}}
[ "$QUICK" = 1 ] && TAG="quick_${STAMP}"
mkdir -p "$OUTDIR" "$DOCDIR"
LOG="$OUTDIR/${TAG}_log.txt"
FAILED=()

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
step() {  # step <nama> <perintah...>
  local name=$1; shift
  say "=== $name"
  if "$@" 2>&1 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" -ne 0 ]; then
    say "!! GAGAL: $name"; FAILED+=("$name")
  fi
}

# --- 0. lingkungan ---------------------------------------------------------------------
ENVF="$OUTDIR/${TAG}_env.txt"
{
  echo "tanggal: $(date -Is)"; echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo tidak-ada-git)"
  echo "git status:"; git status --short 2>/dev/null | head -20
  echo "--- uname"; uname -a; grep -qi microsoft /proc/version 2>/dev/null && echo "WSL: ya" || echo "WSL: tidak"
  echo "--- lscpu"; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|MHz"
  echo "--- memori"; free -h | head -2
  echo "--- gpu"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | head -3
  echo "--- nvcc"; (${NVCC:-nvcc} --version || /usr/local/cuda/bin/nvcc --version) 2>&1 | tail -2
  echo "--- gcc"; g++ --version | head -1
  echo "--- python"; python3 --version; python3 -c "import numpy, PIL; print('numpy', numpy.__version__, 'PIL', PIL.__version__)" 2>&1
  echo "--- daya (laptop: pastikan charger terpasang, mode performa)"
  command -v powercfg.exe >/dev/null 2>&1 && powercfg.exe /getactivescheme 2>&1 | head -2
} > "$ENVF" 2>&1
say "lingkungan dicatat di $ENVF"
CPU=$(lscpu | sed -n 's/^Model name:[ \t]*//p' | head -1)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
LABEL="${CPU}${GPU:+ + $GPU}"
say "mesin: $LABEL"

# --- 1. build --------------------------------------------------------------------------
step "make landcover" make landcover
[ -x build/landcover ] || { say "build/landcover tidak ada; berhenti."; exit 2; }
INFO=$(./build/landcover --info)
HAS_CUDA=$(echo "$INFO" | python3 -c "import json,sys; print(1 if json.load(sys.stdin)['engines'].get('cuda') else 0)")
say "landcover --info: $(echo "$INFO" | tr -d '\n ')"
HAS_BENCH_CUDA=0
if [ "$HAS_CUDA" = 1 ]; then
  if make build/bench_cuda 2>&1 | tee -a "$LOG" | tail -3; [ -x build/bench_cuda ]; then HAS_BENCH_CUDA=1; fi
else
  say "CUDA tidak terhubung di landcover: langkah GPU akan dilewati dan dicatat."
  FAILED+=("CUDA tidak terhubung pada build ini (hasil GPU tidak ada)")
fi

RB="python3 bench/run_bench.py --threads $THREADS --outdir $OUTDIR"
if [ "$QUICK" = 1 ]; then
  S_STRONG=300x300,600x600; S_K=600x600; S_WEAK=100x100; S_VAR=300x300,600x600; KS=2,4,8; KV=4,8; WK=4
else
  S_STRONG=1000x1000,2000x2000,4000x3000,7680x4320; S_K=4000x3000; S_WEAK=2000x1000; S_VAR=1000x1000,4000x3000,7680x4320; KS=2,4,8,16; KV=4,8,16; WK=4,8
fi

# --- 2. langkah pengukuran -------------------------------------------------------------
SUMS=()
[ -f "$OUTDIR/bench_gpu_summary.csv" ] && [ "$QUICK" != 1 ] && SUMS+=("$OUTDIR/bench_gpu_summary.csv")   # data 27 Nov, mesin sama

if [ "$STRONG" = 1 ]; then
  step "strong scaling K=8" $RB --sizes $S_STRONG --k 8 --tag ${TAG}_strong --md $DOCDIR/BENCH_${TAG}_strong.md
  SUMS+=("$OUTDIR/${TAG}_strong_summary.csv")
fi
step "sapuan K ($S_K, K=$KS)" $RB --sizes $S_K --k $KS --tag ${TAG}_ksweep --md $DOCDIR/BENCH_${TAG}_ksweep.md
SUMS+=("$OUTDIR/${TAG}_ksweep_summary.csv")
step "weak scaling omp ($S_WEAK x T, K=$WK)" $RB --weak $S_WEAK --engines omp --k $WK --tag ${TAG}_weak --md $DOCDIR/BENCH_${TAG}_weak.md
SUMS+=("$OUTDIR/${TAG}_weak_summary.csv")

SENT=${SENTINEL:-$(ls data/sentinel*.png 2>/dev/null | tr '\n' ' ')}
if [ -n "${SENT// }" ] && [ "$QUICK" != 1 ]; then
  IMGS=(); for p in $SENT; do IMGS+=(--image "$p"); done
  step "citra Sentinel-2 asli" $RB "${IMGS[@]}" --k 4,8,16 --tag ${TAG}_sentinel --md $DOCDIR/BENCH_${TAG}_sentinel.md
  SUMS+=("$OUTDIR/${TAG}_sentinel_summary.csv")
else
  say "tidak ada PNG Sentinel (data/sentinel*.png); langkah ini dilewati."
  [ "$QUICK" = 1 ] || FAILED+=("Sentinel asli: belum diukur (tidak ada data/sentinel*.png)")
fi

if [ "$HAS_BENCH_CUDA" = 1 ]; then
  step "varian CUDA simple,shared,streams" $RB --sizes $S_VAR --engines "" --k $KV --cuda-variants simple,shared,streams \
      --tag ${TAG}_cudavar --md $DOCDIR/BENCH_${TAG}_cudavar.md
  SUMS+=("$OUTDIR/${TAG}_cudavar_summary.csv")
fi

# --- 3. analisis -----------------------------------------------------------------------
ARGS=(); for s in "${SUMS[@]}"; do [ -s "$s" ] && ARGS+=(--summary "$s"); done
if [ ${#ARGS[@]} -gt 0 ]; then
  step "analisis scaling" python3 bench/analyze_scaling.py "${ARGS[@]}" --label "$LABEL" \
      --md $DOCDIR/BENCH_scaling_${TAG}.md --figdir $DOCDIR/fig/${TAG}
fi

# --- 4. arsip --------------------------------------------------------------------------
ARC="bench_hasil_${TAG}.tar.gz"
FILES=$(ls $OUTDIR/${TAG}_* $DOCDIR/BENCH_${TAG}_* $DOCDIR/BENCH_scaling_${TAG}.md 2>/dev/null; [ -d $DOCDIR/fig/${TAG} ] && find $DOCDIR/fig/${TAG} -type f)
tar czf "$ARC" $FILES
say "arsip: $ARC  (kirim file ini ke Gavra)"

echo; echo "================ RINGKASAN ================"
echo "mesin: $LABEL | tag: $TAG | quick: $QUICK"
if [ ${#FAILED[@]} -eq 0 ]; then echo "semua langkah selesai."; else
  echo "langkah bermasalah / belum diukur:"; for f in "${FAILED[@]}"; do echo "  - $f"; done; fi
grep -l "BEDA" $OUTDIR/${TAG}_*_summary.csv 2>/dev/null | sed 's/^/PERINGATAN hasil BEDA dari acuan di: /'
[ "$QUICK" = 1 ] && echo "MODE QUICK: hasil ini hanya uji skrip, jangan masuk laporan."
exit 0
