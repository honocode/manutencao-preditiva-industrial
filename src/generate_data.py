from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "industrial_synthetic.csv"


def main(n: int = 8000, seed: int = 42, n_assets: int = 20) -> None:
    """Gera leituras sintéticas com identidade e sequência temporal por máquina."""
    if n <= 0 or n_assets <= 0:
        raise ValueError("n e n_assets precisam ser maiores que zero.")

    rng = np.random.default_rng(seed)
    rows_per_asset = int(np.ceil(n / n_assets))
    asset_number = np.repeat(np.arange(n_assets), rows_per_asset)[:n]
    step = np.tile(np.arange(rows_per_asset), n_assets)[:n]
    progress = step / max(rows_per_asset - 1, 1)
    asset_ids = np.array([f"EQ-{number + 1:03d}" for number in asset_number])

    # Cada máquina tem um ponto de operação próprio, e alguns sensores derivam
    # gradualmente ao longo da série para tornar os gráficos de histórico úteis.
    temp_baseline = rng.normal(69, 3.5, n_assets)[asset_number]
    vibration_baseline = rng.normal(0.08, 0.06, n_assets)[asset_number]
    pressure_baseline = rng.normal(6.5, 0.25, n_assets)[asset_number]
    hours_start = rng.integers(500, 4500, n_assets)[asset_number]

    temperature = (
        temp_baseline + 4.5 * progress + rng.normal(0, 8.5, n)
    ).clip(35, 120)
    vibration = rng.lognormal(
        mean=0.05 + vibration_baseline + 0.32 * progress,
        sigma=0.42,
        size=n,
    ).clip(0.1, 9)
    rpm = rng.normal(1500, 190, n).clip(650, 2400)
    pressure = (
        pressure_baseline - 0.25 * progress + rng.normal(0, 0.85, n)
    ).clip(1.5, 12)
    hours = (hours_start + step * 44 + rng.integers(0, 35, n)).clip(1, 24000)

    # Mecanismo artificial: associa sensores e horas de operação à falha.
    logit = (
        -4.0
        + 0.055 * (temperature - 70)
        + 1.05 * (vibration - 1.3)
        + 0.00010 * (hours - 10000)
        + 0.16 * (pressure - 6.5)
    )
    probability = 1 / (1 + np.exp(-logit))
    failure = rng.binomial(1, probability)

    first_reading = pd.Timestamp("2026-01-01T00:00:00Z")
    timestamp = first_reading + pd.to_timedelta(step * 6, unit="h")

    df = pd.DataFrame({
        "timestamp": timestamp,
        "asset_id": asset_ids,
        "temperature_c": temperature.round(2),
        "vibration_mm_s": vibration.round(3),
        "rpm": rpm.round(1),
        "pressure_bar": pressure.round(2),
        "operating_hours": hours.astype(int),
        "failure": failure,
    }).sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Dados sintéticos salvos em: {OUT}")
    print(f"Registros: {len(df):,} | equipamentos: {df.asset_id.nunique()}")
    print(f"Falhas simuladas: {int(df.failure.sum()):,}")
    print(f"Taxa de falha: {df.failure.mean():.2%}")
    print(f"Período: {df.timestamp.min()} até {df.timestamp.max()}")


if __name__ == "__main__":
    main()
