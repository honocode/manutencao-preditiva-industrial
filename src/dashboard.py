from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "predictive_maintenance.joblib"
DATA_PATH = ROOT / "data" / "industrial_synthetic.csv"
METRICS_PATH = ROOT / "reports" / "metrics.json"

st.set_page_config(
    page_title="Industrial Predictive Analytics",
    page_icon="🏭",
    layout="wide"
)

st.title("Industrial Predictive Analytics")
st.caption("Monitoramento demonstrativo de condição operacional e risco de falha")
st.warning(
    "Demonstração educacional com dados sintéticos. "
    "As estimativas não são validadas para máquinas reais."
)

if not MODEL_PATH.exists() or not DATA_PATH.exists() or not METRICS_PATH.exists():
    st.error("Arquivos do modelo ou relatórios não encontrados. Execute generate_data.py e train.py.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]
threshold = float(bundle["threshold"])
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
df = pd.read_csv(DATA_PATH)

st.subheader("Indicadores do conjunto demonstrativo")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Falhas simuladas", f"{int(df.failure.sum()):,}")
k3.metric("Taxa de falha", f"{df.failure.mean():.2%}")
k4.metric("Recall no teste", f"{metrics['recall_failure']:.1%}")

st.divider()
st.subheader("Simulador de condição operacional")
st.write("Ajuste os parâmetros para consultar a saída do modelo.")

left, mid, right = st.columns(3)
with left:
    temp = st.slider("Temperatura (°C)", 35.0, 120.0, 72.0)
    vibration = st.slider("Vibração (mm/s)", 0.1, 9.0, 1.5)
with mid:
    rpm = st.slider("Rotação (RPM)", 650.0, 2400.0, 1500.0)
    pressure = st.slider("Pressão (bar)", 1.5, 12.0, 6.5)
with right:
    hours = st.number_input("Horas de operação", 1, 24000, 8000)

sample = pd.DataFrame([{
    "temperature_c": temp,
    "vibration_mm_s": vibration,
    "rpm": rpm,
    "pressure_bar": pressure,
    "operating_hours": hours
}])[features]

probability = float(model.predict_proba(sample)[0, 1])
if probability >= threshold:
    risk_label = "Alto — acima do limiar do modelo"
    st.error(risk_label)
elif probability >= threshold * 0.5:
    risk_label = "Moderado — acompanhar condição"
    st.warning(risk_label)
else:
    risk_label = "Baixo — abaixo da faixa de alerta"
    st.success(risk_label)

p1, p2 = st.columns([1, 2])
p1.metric("Score estimado pelo modelo", f"{probability:.1%}")
p2.metric("Limiar de classificação", f"{threshold:.1%}")
st.progress(min(max(probability, 0.0), 1.0))
st.caption("O score é produzido por um modelo treinado em dados artificiais; não equivale a risco operacional validado.")

st.divider()
st.subheader("Desempenho do modelo — conjunto de teste")
m1, m2, m3, m4 = st.columns(4)
m1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
m2.metric("Average Precision", f"{metrics['average_precision']:.3f}")
m3.metric("Precision (falha)", f"{metrics['precision_failure']:.1%}")
m4.metric("F1-score (falha)", f"{metrics['f1_failure']:.1%}")

c1, c2 = st.columns(2)
with c1:
    st.image(str(ROOT / "reports" / "confusion_matrix.png"), caption="Matriz de confusão — teste")
with c2:
    st.image(str(ROOT / "reports" / "feature_importance.png"), caption="Importância das variáveis")

st.divider()
st.subheader("Amostra dos dados")
st.dataframe(df.head(15), use_container_width=True)
