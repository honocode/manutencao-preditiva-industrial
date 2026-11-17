from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "industrial_synthetic.csv"

def main(n: int = 8000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    temperature = rng.normal(70, 10, n).clip(35, 120)
    vibration = rng.lognormal(mean=0.12, sigma=0.50, size=n).clip(0.1, 9)
    rpm = rng.normal(1500, 190, n).clip(650, 2400)
    pressure = rng.normal(6.5, 1.0, n).clip(1.5, 12)
    hours = rng.integers(1, 24001, n)

    # Mecanismo artificial: cria associação didática entre sensores e falha.
    logit = (
        -4.0
        + 0.055 * (temperature - 70)
        + 1.05 * (vibration - 1.3)
        + 0.00010 * (hours - 10000)
        + 0.16 * (pressure - 6.5)
    )
    probability = 1 / (1 + np.exp(-logit))
    failure = rng.binomial(1, probability)

    df = pd.DataFrame({
        "temperature_c": temperature.round(2),
        "vibration_mm_s": vibration.round(3),
        "rpm": rpm.round(1),
        "pressure_bar": pressure.round(2),
        "operating_hours": hours,
        "failure": failure,
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Dados sintéticos salvos em: {OUT}")
    print(f"Registros: {len(df):,}")
    print(f"Falhas simuladas: {int(df.failure.sum()):,}")
    print(f"Taxa de falha: {df.failure.mean():.2%}")
    print("\nDistribuição do alvo:")
    print(df["failure"].value_counts().sort_index())

if __name__ == "__main__":
    main()
