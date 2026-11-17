from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve, roc_auc_score
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "industrial_synthetic.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

FEATURES = [
    "temperature_c", "vibration_mm_s", "rpm",
    "pressure_bar", "operating_hours"
]
TARGET = "failure"

TARGET_RECALL = 0.80

def choose_threshold(y_true, probabilities, target_recall=TARGET_RECALL):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    # Prioriza encontrar falhas; entre limiares que atingem a meta,
    # escolhe o mais alto para reduzir alarmes falsos.
    eligible = np.flatnonzero(recall[:-1] >= target_recall)
    if len(eligible) == 0:
        return float(thresholds[int(np.argmax(recall[:-1]))])
    return float(thresholds[eligible[-1]])

def main():
    if not DATA.exists():
        raise FileNotFoundError("Execute primeiro: python3 src/generate_data.py")

    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    valid_prob = model.predict_proba(X_valid)[:, 1]
    threshold = choose_threshold(y_valid, valid_prob)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    report = classification_report(
        y_test, test_pred, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    metrics = {
        "threshold": threshold,
        "threshold_policy": "Maior limiar que atinge recall alvo na validação",
        "target_recall": TARGET_RECALL,
        "validation_recall_at_threshold": float(
            np.mean((valid_prob >= threshold)[y_valid.to_numpy() == 1])
        ),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "test_prevalence": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "average_precision": float(average_precision_score(y_test, test_prob)),
        "precision_failure": float(report.get("1", {}).get("precision", 0)),
        "recall_failure": float(report.get("1", {}).get("recall", 0)),
        "f1_failure": float(report.get("1", {}).get("f1-score", 0)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_valid)),
        "test_rows": int(len(y_test)),
        "test_failures": int(y_test.sum()),
        "note": "Avaliação em dados sintéticos; não representa desempenho industrial real."
    }

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURES, "threshold": threshold},
        MODEL_DIR / "predictive_maintenance.joblib"
    )
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Sem falha", "Falha"]
    ).plot(values_format="d")
    plt.title("Matriz de confusão — teste sintético")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=170)
    plt.close()

    importance = pd.Series(
        model.feature_importances_, index=FEATURES
    ).sort_values()
    ax = importance.plot(kind="barh", title="Importância das variáveis")
    ax.set_xlabel("Importância relativa")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_importance.png", dpi=170)
    plt.close()

    print(f"Recall alvo na validação: {TARGET_RECALL:.0%}")
    print(f"Limiar selecionado na validação: {threshold:.4f}")
    print(f"Falhas detectadas/perdidas no teste: {tp}/{fn}; alarmes falsos: {fp}")
    print(f"ROC-AUC (teste): {metrics['roc_auc']:.3f}")
    print(f"Average Precision (teste): {metrics['average_precision']:.3f}")
    print("\nRelatório de classificação:")
    print(classification_report(y_test, test_pred, zero_division=0))
    print("Matriz de confusão [ [TN, FP], [FN, TP] ]:")
    print(cm)
    print("\nModelo e relatórios salvos em models/ e reports/.")

if __name__ == "__main__":
    main()
