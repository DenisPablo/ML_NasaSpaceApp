#!/usr/bin/env bash
# Corre todas las mediciones en orden y guarda la salida en resultados/.
# Uso (desde la raíz del repo):  bash analisis/veredicto-armonizacion/run_all.sh [python]
set -euo pipefail
PY="${1:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$HERE/_trabajo"; mkdir -p "$WORK" "$HERE/resultados"
cd "$WORK"
for s in m0_inspect m1_repro m2_factor m3_cv m4_oper m4b_gaps m5_conf m6_anom m7_conf m8_orbit m9_extra; do
  echo ">> $s"
  PYTHONWARNINGS=ignore "$PY" "$HERE/scripts/$s.py" > "$HERE/resultados/$s.txt"
done
echo "Listo: resultados en $HERE/resultados/"
