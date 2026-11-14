#!/usr/bin/env bash
# Empaqueta una versión de la clasificación para publicarla como Release de GitHub.
#
# Uso:
#   scripts/empaquetar_version.sh <carpeta-de-versión> [carpeta-de-salida]
#
# Ejemplo:
#   scripts/empaquetar_version.sh data/serving_api/20261114T171326
#   → dist/serving_api_20261114T171326.tar.zst   (o .tar.gz si no hay zstd)
#   → dist/serving_api_20261114T171326.sha256
#
# El archivo contiene la carpeta <versión>/ en su raíz: al descomprimirlo dentro de
# /var/data/releases/ queda /var/data/releases/<versión>/. El .sha256 usa el nombre del
# archivo sin ruta, para verificarlo con `sha256sum -c` (Linux) o `shasum -a 256 -c` (macOS)
# desde la carpeta donde estén los dos archivos.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Uso: $0 <carpeta-de-versión> [carpeta-de-salida (por defecto: dist)]" >&2
  exit 1
fi

src="${1%/}"
out_dir="${2:-dist}"

if [[ ! -d "$src" ]]; then
  echo "No existe la carpeta de versión: $src" >&2
  exit 1
fi
if [[ ! -f "$src/metadata.json" ]]; then
  echo "La carpeta no parece una versión completa (falta metadata.json): $src" >&2
  exit 1
fi

# Si se pasa el enlace "current", se resuelve a la carpeta real de la versión.
if [[ -L "$src" ]]; then
  src="$(cd "$(dirname "$src")" && cd "$(readlink "$(basename "$src")")" && pwd)"
fi
src="$(cd "$src" && pwd)"
version="$(basename "$src")"
parent="$(dirname "$src")"

mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"

# Sin metadatos extendidos de macOS (evita archivos ._* y avisos al descomprimir en Linux).
export COPYFILE_DISABLE=1
tar_opts=()
if tar --version 2>/dev/null | grep -q bsdtar; then
  tar_opts+=(--no-xattrs)
fi

if command -v zstd >/dev/null 2>&1; then
  archive="serving_api_${version}.tar.zst"
  tar "${tar_opts[@]+"${tar_opts[@]}"}" -C "$parent" -cf - "$version" | zstd -q -19 -T0 -o "$out_dir/$archive" -f
else
  archive="serving_api_${version}.tar.gz"
  echo "Aviso: zstd no está instalado; se usa gzip." >&2
  tar "${tar_opts[@]+"${tar_opts[@]}"}" -C "$parent" -cf - "$version" | gzip -9 > "$out_dir/$archive"
fi

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$out_dir" && sha256sum "$archive" > "serving_api_${version}.sha256")
else
  (cd "$out_dir" && shasum -a 256 "$archive" > "serving_api_${version}.sha256")
fi

size="$(du -h "$out_dir/$archive" | cut -f1)"
echo "Listo:"
echo "  $out_dir/$archive ($size)"
echo "  $out_dir/serving_api_${version}.sha256"
