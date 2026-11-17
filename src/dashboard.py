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
    page_title="Manutenção Preditiva | Monitoramento",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root { --ink:#152b3c; --muted:#667b89; --teal:#087e83; --line:#e3ebef; --paper:#f5f8fa; }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
      .stApp { background:var(--paper); }
      [data-testid="stHeader"] { background:rgba(245,248,250,.88); }
      [data-testid="stSidebar"] { background:#102b3b; }
      [data-testid="stSidebar"] * { color:#f2f7f8 !important; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#b8cbd2 !important; }
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#eaf2f4 !important; }
      [data-testid="stSidebar"] [data-baseweb="select"] * { color:#173345 !important; }
      .block-container { padding-top:2.1rem; padding-bottom:3rem; max-width:1440px; }
      .hero { padding:1.55rem 1.8rem; border-radius:18px; color:white; background:linear-gradient(115deg,#102b3b 0%,#17485a 65%,#087e83 100%); margin-bottom:1rem; box-shadow:0 10px 30px rgba(16,43,59,.12); }
      .hero-kicker { font-size:.74rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; color:#9ee0d8; margin:0 0 .45rem 0; }
      .hero h1 { font-family:Manrope,sans-serif; color:white; font-size:2rem; margin:0; letter-spacing:-.03em; }
      .hero-sub { color:#d2e4e8; margin:.45rem 0 0 0; font-size:.98rem; }
      .status-pill { display:inline-block; border:1px solid rgba(255,255,255,.28); border-radius:999px; padding:.32rem .65rem; font-size:.73rem; font-weight:700; letter-spacing:.04em; color:#d8f4ed; background:rgba(255,255,255,.08); }
      .section-title { font-family:Manrope,sans-serif; font-size:1.15rem; font-weight:750; color:var(--ink); margin:.7rem 0 .15rem 0; }
      .section-note { color:var(--muted); font-size:.88rem; margin:0 0 .85rem 0; }
      div[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:14px; padding:1rem 1.05rem; box-shadow:0 3px 12px rgba(20,48,65,.035); }
      div[data-testid="stMetricLabel"] p { color:#627887; font-size:.82rem; font-weight:600; }
      div[data-testid="stMetricValue"] { color:var(--ink); font-family:Manrope,sans-serif; }
      div[data-testid="stMetricDelta"] { font-size:.76rem; }
      .info-card { padding:1rem 1.15rem; border:1px solid var(--line); border-radius:14px; background:white; min-height:115px; }
      .info-label { color:var(--muted); font-size:.77rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; margin-bottom:.35rem; }
      .info-value { color:var(--ink); font-family:Manrope,sans-serif; font-size:1.6rem; line-height:1.2; font-weight:800; }
      .info-detail { color:var(--muted); font-size:.82rem; margin-top:.35rem; }
      .stTabs [data-baseweb="tab-list"] { gap:.45rem; border-bottom:1px solid var(--line); }
      .stTabs [data-baseweb="tab"] { background:transparent; border-radius:8px 8px 0 0; padding:.65rem 1rem; }
      .stTabs [aria-selected="true"] { color:var(--teal); }
      div[data-testid="stAlert"] { border-radius:12px; }
      .footnote { color:#738692; font-size:.78rem; padding-top:.6rem; }
      hr { border-color:var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists() or not DATA_PATH.exists() or not METRICS_PATH.exists():
    st.error("Arquivos do modelo ou dos relatórios não encontrados. Execute primeiro `python3 src/generate_data.py` e `python3 src/train.py`.")
    st.stop()

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

@st.cache_data
def load_frame(path):
    return pd.read_csv(path)

bundle = load_bundle(str(MODEL_PATH))
model = bundle["model"]
features = bundle["features"]
threshold = float(bundle["threshold"])
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
df = load_frame(str(DATA_PATH))

with st.sidebar:
    st.markdown("<div style='font-size:1.4rem;font-weight:800;font-family:Manrope,sans-serif'>🏭  INDÚSTRIA</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;margin-top:.2rem'>CENTRO DE MONITORAMENTO</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### Simulador de condição")
    st.markdown("Ajuste os parâmetros para consultar a estimativa do modelo.")
    temp = st.slider("Temperatura · °C", 35.0, 120.0, 72.0, 0.5)
    vibration = st.slider("Vibração · mm/s", 0.1, 9.0, 1.5, 0.1)
    rpm = st.slider("Rotação · RPM", 650.0, 2400.0, 1500.0, 10.0)
    pressure = st.slider("Pressão · bar", 1.5, 12.0, 6.5, 0.1)
    hours = st.number_input("Horas de operação", 1, 24000, 8000, step=100)
    st.divider()
    st.caption("Modelo demonstrativo · dados sintéticos")

sample = pd.DataFrame([{
    "temperature_c": temp,
    "vibration_mm_s": vibration,
    "rpm": rpm,
    "pressure_bar": pressure,
    "operating_hours": hours,
}])[features]
probability = float(model.predict_proba(sample)[0, 1])

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Painel de confiabilidade</div>
      <h1>Manutenção preditiva</h1>
      <p class="hero-sub">Visão demonstrativa da condição dos equipamentos e do desempenho do modelo.</p>
      <span class="status-pill">●  AMBIENTE DE DEMONSTRAÇÃO</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning("Este painel usa dados sintéticos para fins educacionais. As estimativas não foram validadas para máquinas reais.", icon="⚠️")

st.markdown('<div class="section-title">Visão geral da base</div>', unsafe_allow_html=True)
st.markdown('<div class="section-note">Indicadores do conjunto usado nesta demonstração.</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Registros analisados", f"{len(df):,}")
k2.metric("Falhas simuladas", f"{int(df.failure.sum()):,}")
k3.metric("Taxa de falha", f"{df.failure.mean():.1%}")
k4.metric("Recall no teste", f"{metrics['recall_failure']:.1%}", help="Proporção das falhas do conjunto de teste que o modelo identificou.")

st.markdown("<br>", unsafe_allow_html=True)
tab_overview, tab_simulator, tab_data = st.tabs(["Desempenho", "Simulador", "Dados e método"])

with tab_overview:
    st.markdown('<div class="section-title">Desempenho do modelo</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Resultados em um conjunto de teste separado da etapa usada para escolher o limiar.</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    b.metric("Average Precision", f"{metrics['average_precision']:.3f}")
    c.metric("Precisão · falhas", f"{metrics['precision_failure']:.1%}")
    d.metric("F1-score · falhas", f"{metrics['f1_failure']:.1%}")

    st.markdown("<br>", unsafe_allow_html=True)
    x1, x2, x3 = st.columns(3)
    x1.metric("Falhas detectadas", metrics["true_positives"])
    x2.metric("Falhas não detectadas", metrics["false_negatives"])
    x3.metric("Alertas falsos", metrics["false_positives"])
    st.info(
        f"Limiar de alerta: **{threshold:.1%}**. Foi escolhido na validação para buscar recall de "
        f"{metrics['target_recall']:.0%}; no teste, o recall observado foi de {metrics['recall_failure']:.1%}. "
        "A meta é apenas demonstrativa e pode gerar muitos alertas falsos.",
        icon="ℹ️",
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(ROOT / "reports" / "confusion_matrix.png"), caption="Matriz de confusão · conjunto de teste", use_container_width=True)
    with col2:
        st.image(str(ROOT / "reports" / "feature_importance.png"), caption="Importância das variáveis no modelo", use_container_width=True)

with tab_simulator:
    st.markdown('<div class="section-title">Estimativa para a condição selecionada</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Os controles ficam na barra lateral. O resultado é uma pontuação do modelo, não um diagnóstico.</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        st.metric("Score estimado", f"{probability:.1%}")
        st.metric("Limiar atual", f"{threshold:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
    with right:
        if probability >= threshold:
            st.error("Acima do limiar de alerta do modelo", icon="🚨")
        elif probability >= threshold * 0.5:
            st.warning("Faixa intermediária do simulador", icon="👀")
        else:
            st.success("Abaixo do limiar de alerta", icon="✅")
        st.markdown("#### Parâmetros selecionados")
        params = pd.DataFrame({"Parâmetro": ["Temperatura", "Vibração", "Rotação", "Pressão", "Horas de operação"], "Valor": [f"{temp:.1f} °C", f"{vibration:.1f} mm/s", f"{rpm:.0f} RPM", f"{pressure:.1f} bar", f"{hours:,} h"]})
        st.dataframe(params, hide_index=True, use_container_width=True)
    st.caption("O score é produzido por um modelo treinado em dados artificiais; não equivale a risco operacional validado.")

with tab_data:
    st.markdown('<div class="section-title">Dados e método</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Resumo da base e amostra dos registros utilizados na demonstração.</div>', unsafe_allow_html=True)
    info1, info2, info3 = st.columns(3)
    info1.metric("Amostras de treino", f"{metrics['train_rows']:,}")
    info2.metric("Amostras de validação", f"{metrics['validation_rows']:,}")
    info3.metric("Amostras de teste", f"{metrics['test_rows']:,}")
    st.markdown("#### Amostra da base sintética")
    st.dataframe(df.head(20), hide_index=True, use_container_width=True)
    st.markdown('<div class="footnote">Os dados foram gerados artificialmente. Métricas e scores servem apenas para demonstração e não devem orientar manutenção real.</div>', unsafe_allow_html=True)
