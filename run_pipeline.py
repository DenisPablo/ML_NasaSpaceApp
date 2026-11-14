"""Interactive CLI runner for the NASA FIRMS MODIS x VIIRS harmonization pipeline."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Callable

from aggregate_daily import run_aggregation_pipeline
from clean_firms import run_cleaning_pipeline
from predict_harmonized import run_harmonized_inference
from train_hurdle import train_models_with_dual_validation

DEFAULT_RAW_DIR = Path("Data-Argentina")
DEFAULT_CLEAN_PARQUET = Path("data/clean/firms_argentina_clean.parquet")
DEFAULT_DAILY_DIR = Path("data/processed/firms_argentina_daily")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_SERVING_DIR = Path("data/serving/firms_argentina_harmonized")


def get_pipeline_status(
    clean_parquet: Path = DEFAULT_CLEAN_PARQUET,
    daily_dir: Path = DEFAULT_DAILY_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    serving_dir: Path = DEFAULT_SERVING_DIR,
) -> dict[str, bool]:
    """Check whether each pipeline stage artifact exists on disk."""
    has_daily = daily_dir.exists() and any(daily_dir.glob("year=*/data.parquet"))
    has_models = (models_dir / "n20_count_tweedie.txt").exists() and (
        models_dir / "n20_frp_tweedie.txt"
    ).exists()
    has_serving = serving_dir.exists() and any(serving_dir.glob("year=*/data.parquet"))
    return {
        "phase1_clean": clean_parquet.exists(),
        "phase2_daily": has_daily,
        "phase3_models": has_models,
        "phase4_serving": has_serving,
    }


def format_menu(status: dict[str, bool], region: str) -> str:
    """Render the interactive pipeline menu with live artifact status."""
    mark = lambda ok: "OK" if ok else "PENDIENTE"
    return (
        "\n=== Pipeline MODIS x VIIRS (Escala Argentina) ===\n"
        f"Region actual: {region}\n"
        f"  [1] Fase 1 - Limpieza CSVs -> Parquet limpio      [{mark(status['phase1_clean'])}]\n"
        f"  [2] Fase 2 - Matriz diaria dispersa (Aqua/N20)    [{mark(status['phase2_daily'])}]\n"
        f"  [3] Fase 3 - Entrenamiento Tweedie + Validacion   [{mark(status['phase3_models'])}]\n"
        f"  [4] Fase 4 - Inferencia y Parquet Serving         [{mark(status['phase4_serving'])}]\n"
        "  [5] Ejecutar Pipeline Completo (Fases 1 -> 4)\n"
        "  [6] Correr Tests Unitarios (pytest)\n"
        "  [7] Cambiar region (argentina / san_luis)\n"
        "  [0] Salir\n"
    )


def _run_phase1(raw_dir: Path, clean_parquet: Path, region: str, out: Callable[[str], None]) -> None:
    stats = run_cleaning_pipeline(input_dir=raw_dir, output_parquet=clean_parquet, region=region)
    out(
        f"Fase 1 OK en {stats.elapsed_seconds:.2f}s | Archivos: {stats.files_processed} | "
        f"Leidas: {stats.rows_read:,} | Finales: {stats.rows_deduped:,}"
    )


def _run_phase2(clean_parquet: Path, daily_dir: Path, out: Callable[[str], None]) -> None:
    stats = run_aggregation_pipeline(input_parquet=clean_parquet, output_parquet=daily_dir)
    out(
        f"Fase 2 OK en {stats.elapsed_seconds:.2f}s | Celdas: {stats.active_cells:,} | "
        f"Filas activas: {stats.total_rows:,} | MODIS>0: {stats.modis_active_rows:,} | N20>0: {stats.n20_active_rows:,}"
    )


def _run_phase3(daily_dir: Path, models_dir: Path, out: Callable[[str], None]) -> None:
    stats = train_models_with_dual_validation(input_path=daily_dir, models_dir=models_dir)
    out(
        f"Fase 3 OK en {stats.elapsed_seconds:.2f}s | Activas: {stats.train_rows:,} | "
        f"Temp R2 Count={stats.temporal_r2_count:.4f} (Base 5.5x={stats.temporal_r2_baseline:.4f}) | "
        f"LOBBO R2 Count={stats.lobbo_r2_count:.4f}, FRP={stats.lobbo_r2_frp:.4f}"
    )


def _run_phase4(daily_dir: Path, models_dir: Path, serving_dir: Path, out: Callable[[str], None]) -> None:
    stats = run_harmonized_inference(input_path=daily_dir, models_dir=models_dir, output_path=serving_dir)
    out(
        f"Fase 4 OK en {stats.elapsed_seconds:.2f}s | Filas serving: {stats.total_rows:,} | "
        f"Estimadas: {stats.estimated_rows:,} | Observadas N20: {stats.observed_rows:,}"
    )


def run_interactive_cli(
    raw_dir: Path = DEFAULT_RAW_DIR,
    clean_parquet: Path = DEFAULT_CLEAN_PARQUET,
    daily_dir: Path = DEFAULT_DAILY_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    serving_dir: Path = DEFAULT_SERVING_DIR,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    test_runner: Callable[[], int] | None = None,
) -> int:
    """Run the interactive menu loop until the user exits."""
    region = "argentina"
    while True:
        status = get_pipeline_status(clean_parquet, daily_dir, models_dir, serving_dir)
        output_fn(format_menu(status, region))
        try:
            choice = input_fn("Selecciona una opcion: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nSaliendo del pipeline interactivo.")
            return 0

        if choice == "0":
            output_fn("Hasta luego.")
            return 0
        try:
            if choice == "1":
                _run_phase1(raw_dir, clean_parquet, region, output_fn)
            elif choice == "2":
                _run_phase2(clean_parquet, daily_dir, output_fn)
            elif choice == "3":
                _run_phase3(daily_dir, models_dir, output_fn)
            elif choice == "4":
                _run_phase4(daily_dir, models_dir, serving_dir, output_fn)
            elif choice == "5":
                _run_phase1(raw_dir, clean_parquet, region, output_fn)
                _run_phase2(clean_parquet, daily_dir, output_fn)
                _run_phase3(daily_dir, models_dir, output_fn)
                _run_phase4(daily_dir, models_dir, serving_dir, output_fn)
            elif choice == "6":
                rc = (
                    test_runner()
                    if test_runner is not None
                    else subprocess.call([sys.executable, "-m", "pytest", "-v"])
                )
                output_fn(f"Tests finalizados con codigo: {rc}")
            elif choice == "7":
                region = "san_luis" if region == "argentina" else "argentina"
                output_fn(f"Region cambiada a: {region}")
            else:
                output_fn(f"Opcion invalida: '{choice}'. Elegi entre 0 y 7.")
        except Exception as exc:
            output_fn(f"Error ejecutando opcion {choice}: {exc}")


if __name__ == "__main__":
    raise SystemExit(run_interactive_cli())
