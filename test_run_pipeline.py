"""Tests for the interactive pipeline CLI menu and routing."""

from pathlib import Path

from run_pipeline import format_menu, get_pipeline_status, run_interactive_cli


def test_status_and_menu_formatting(tmp_path: Path) -> None:
    clean_p = tmp_path / "clean.parquet"
    daily_d = tmp_path / "daily"
    models_d = tmp_path / "models"
    serving_d = tmp_path / "serving"

    status = get_pipeline_status(clean_p, daily_d, models_d, serving_d)
    assert status == {
        "phase1_clean": False,
        "phase2_daily": False,
        "phase3_models": False,
        "phase4_serving": False,
    }

    clean_p.write_text("ok")
    (daily_d / "year=2024").mkdir(parents=True)
    (daily_d / "year=2024" / "data.parquet").write_text("ok")
    status2 = get_pipeline_status(clean_p, daily_d, models_d, serving_d)
    assert status2["phase1_clean"] is True
    assert status2["phase2_daily"] is True

    menu = format_menu(status2, "argentina")
    assert "[OK]" in menu
    assert "[PENDIENTE]" in menu
    assert "Region actual: argentina" in menu


def test_interactive_cli_region_toggle_tests_and_exit(tmp_path: Path) -> None:
    inputs = iter(["9", "7", "6", "0"])
    outputs: list[str] = []
    test_calls = 0

    def _fake_test() -> int:
        nonlocal test_calls
        test_calls += 1
        return 0

    rc = run_interactive_cli(
        raw_dir=tmp_path / "raw",
        clean_parquet=tmp_path / "clean.parquet",
        daily_dir=tmp_path / "daily",
        models_dir=tmp_path / "models",
        serving_dir=tmp_path / "serving",
        input_fn=lambda _: next(inputs),
        output_fn=outputs.append,
        test_runner=_fake_test,
    )

    assert rc == 0
    assert test_calls == 1
    joined = "\n".join(outputs)
    assert "Opcion invalida: '9'" in joined
    assert "Region cambiada a: san_luis" in joined
    assert "Tests finalizados con codigo: 0" in joined
