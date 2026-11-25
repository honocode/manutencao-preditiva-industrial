from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "predictive_maintenance.joblib"
DATA_PATH = ROOT / "data" / "industrial_synthetic.csv"
METRICS_PATH = ROOT / "reports" / "metrics.json"
LOGO_PATH = ROOT / "images" / "honocode_logo.png"

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
      :root { --ink:#1F2937; --muted:#667b89; --teal:#3B82F6; --primary-color:#3B82F6; --line:#e3ebef; --paper:#F3F4F6; }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
      .stApp { background:var(--paper); }
      [data-testid="stHeader"] { background:rgba(243,244,246,.88); }
      [data-testid="stHeader"] button, [data-testid="stHeader"] button *, [data-testid="stSidebarCollapsedControl"] button, [data-testid="stSidebarCollapsedControl"] button * { color:#1F2937 !important; fill:#1F2937 !important; opacity:1 !important; }
      [data-testid="stHeader"] button:hover, [data-testid="stSidebarCollapsedControl"] button:hover { background:#E5E7EB !important; }
      [data-testid="stSidebar"] { background:#172554; }
      [data-testid="stSidebar"] * { color:#f2f7f8 !important; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#b8cbd2 !important; }
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#eaf2f4 !important; }
      [data-testid="stSidebar"] [data-baseweb="select"] * { color:#173345 !important; }
      .block-container { padding-top:2.1rem; padding-bottom:3rem; max-width:1440px; }
      .hero { padding:1.55rem 1.8rem; border-radius:18px; color:white; background:#1E40AF; margin-bottom:1rem; box-shadow:0 10px 30px rgba(16,43,59,.12); }
      .hero-kicker { font-size:.74rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; color:#bfdbfe; margin:0 0 .45rem 0; }
      .hero h1 { font-family:Manrope,sans-serif; color:white; font-size:2rem; margin:0; letter-spacing:-.03em; }
      .hero-sub { color:#d2e4e8; margin:.45rem 0 0 0; font-size:.98rem; }
      .section-title { font-family:Manrope,sans-serif; font-size:1.15rem; font-weight:750; color:var(--ink); margin:.7rem 0 .15rem 0; }
      .section-note { color:var(--muted); font-size:.88rem; margin:0 0 .85rem 0; }
      div[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:14px; padding:1rem 1.05rem; box-shadow:0 3px 12px rgba(20,48,65,.035); }
      html body .stApp div[data-testid="stMetric"] div[data-testid="stMetricLabel"], html body .stApp div[data-testid="stMetric"] div[data-testid="stMetricLabel"] * { color:#1F2937 !important; -webkit-text-fill-color:#1F2937 !important; opacity:1 !important; }
      html body .stApp div[data-testid="stMetric"] div[data-testid="stMetricLabel"] p, html body .stApp div[data-testid="stMetric"] div[data-testid="stMetricLabel"] label { color:#1F2937 !important; -webkit-text-fill-color:#1F2937 !important; font-size:.88rem; font-weight:700; opacity:1 !important; }
      .section-note { color:#4B5563 !important; opacity:1 !important; }
      div[data-testid="stMetricValue"] { color:var(--ink); font-family:Manrope,sans-serif; }
      div[data-testid="stMetricDelta"] { font-size:.76rem; }
      .info-card { padding:1rem 1.15rem; border:1px solid var(--line); border-radius:14px; background:white; min-height:115px; }
      .info-label { color:var(--muted); font-size:.77rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; margin-bottom:.35rem; }
      .info-value { color:var(--ink); font-family:Manrope,sans-serif; font-size:1.6rem; line-height:1.2; font-weight:800; }
      .info-detail { color:var(--muted); font-size:.82rem; margin-top:.35rem; }
      .stTabs [data-baseweb="tab-list"], [role="tablist"] { gap:.55rem !important; border-bottom:0 !important; align-items:center; }
      [data-baseweb="tab-list"] { border-bottom:0 !important; box-shadow:none !important; gap:.55rem !important; }
      [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none !important; visibility:hidden !important; height:0 !important; border:0 !important; background:none !important; }
      [role="tablist"]::before, [role="tablist"]::after, [data-baseweb="tab-list"]::before, [data-baseweb="tab-list"]::after, [role="tab"]::before, [role="tab"]::after { display:none !important; content:none !important; border:0 !important; box-shadow:none !important; }
      .stTabs [data-baseweb="tab"], [role="tab"] { box-sizing:border-box !important; width:max-content !important; min-width:max-content !important; white-space:nowrap !important; padding:.6rem .9rem !important; margin:0 !important; border:0 !important; border-radius:0 !important; background:transparent !important; color:#374151 !important; -webkit-text-fill-color:#374151 !important; opacity:1 !important; font-size:.95rem !important; font-weight:700 !important; line-height:1.35 !important; }
      .stTabs [data-baseweb="tab"] *, [role="tab"] * { color:#374151 !important; -webkit-text-fill-color:#374151 !important; opacity:1 !important; }
      .stTabs [data-baseweb="tab"][aria-selected="true"], [role="tab"][aria-selected="true"] { background:#3B82F6 !important; background-image:none !important; border:0 !important; border-bottom:0 !important; border-radius:0 !important; box-shadow:none !important; color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; }
      [role="tab"][aria-selected="true"] { position:relative !important; z-index:3 !important; }
      [role="tab"][aria-selected="true"]::after { content:"" !important; display:block !important; position:absolute !important; left:0 !important; right:0 !important; bottom:-2px !important; height:4px !important; background:#3B82F6 !important; border:0 !important; box-shadow:none !important; z-index:5 !important; }
      .stTabs [data-baseweb="tab"][aria-selected="true"] *, [role="tab"][aria-selected="true"] * { color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; }
      .stTabs [data-baseweb="tab"][aria-selected="true"] *, [role="tab"][aria-selected="true"] * { border:0 !important; box-shadow:none !important; background-image:none !important; }
      .stTabs [data-baseweb="tab"]:focus-visible, [role="tab"]:focus-visible { outline:3px solid #93C5FD !important; outline-offset:2px; }
      div[data-testid="stAlert"] { border-radius:12px; color:#1F2937; background:#FFFFFF !important; border:1px solid #E5E7EB; }
      div[data-testid="stAlert"] p { color:#1F2937 !important; }
      div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] { color:#1F2937 !important; background:#FFFFFF !important; }
      .notice-warning { display:flex; align-items:flex-start; gap:.75rem; margin:0 0 1rem; padding:.9rem 1rem; color:#1F2937; background:#FFFFFF; border:1px solid #E5E7EB; border-left:4px solid #D97706; border-radius:12px; }
      .notice-warning-icon, .notice-info-icon { flex:0 0 auto; line-height:1.4; }
      .notice-warning-text, .notice-info-text { color:#1F2937; line-height:1.5; }
      .notice-info { display:flex; align-items:flex-start; gap:.75rem; margin:0 0 1rem; padding:.9rem 1rem; color:#1F2937; background:#FFFFFF; border:1px solid #E5E7EB; border-left:4px solid #1E40AF; border-radius:12px; }
      .footnote { color:#738692; font-size:.78rem; padding-top:.6rem; }
      hr { border-color:var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists() or not DATA_PATH.exists() or not METRICS_PATH.exists():
    st.error("Arquivos do modelo ou dos relatórios não encontrados. Execute primeiro `python3 src/generate_data.py` e `python3 src/train.py`.")
    st.stop()

def file_version(path):
    """Inclui a versão do arquivo na chave do cache para evitar dados antigos."""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size

@st.cache_resource
def load_bundle(path, version):
    return joblib.load(path)

@st.cache_data
def load_frame(path, version):
    return pd.read_csv(path, parse_dates=["timestamp"])

bundle = load_bundle(str(MODEL_PATH), file_version(MODEL_PATH))
model = bundle["model"]
features = bundle["features"]
threshold = float(bundle["threshold"])
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
df = load_frame(str(DATA_PATH), file_version(DATA_PATH))

required_history_columns = {"timestamp", "asset_id"}
if not required_history_columns.issubset(df.columns):
    st.error("A base ainda não tem histórico por equipamento. Execute `python3 src/generate_data.py` e depois `python3 src/train.py`.")
    st.stop()

asset_ids = sorted(df["asset_id"].dropna().unique().tolist())
if not asset_ids:
    st.error("Nenhum equipamento foi encontrado na base.")
    st.stop()

with st.sidebar:
    logo_left, logo_center, logo_right = st.columns([1, 2, 1], gap="small")
    with logo_center:
        st.image(str(LOGO_PATH), width=145)
    st.divider()
    st.markdown("### Equipamento selecionado")
    selected_asset = st.selectbox("Máquina", asset_ids, index=0)
    st.markdown("### Simulador de condição")
    st.markdown("Ajuste os parâmetros para consultar a estimativa do modelo.")
    temp = st.slider("Temperatura · °C", 35.0, 120.0, 72.0, 0.5)
    vibration = st.slider("Vibração · mm/s", 0.1, 9.0, 1.5, 0.1)
    rpm = st.slider("Rotação · RPM", 650.0, 2400.0, 1500.0, 10.0)
    pressure = st.slider("Pressão · bar", 1.5, 12.0, 6.5, 0.1)
    hours = st.number_input("Horas de operação", 1, 24000, 8000, step=100)
    st.divider()
    st.caption("Modelo demonstrativo · dados sintéticos")
    if st.button("⟳  Atualizar dados e modelo", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.caption("Use este botão após regenerar os dados ou treinar um novo modelo.")

asset_history = (
    df.loc[df["asset_id"] == selected_asset]
    .sort_values("timestamp")
    .tail(120)
    .copy()
)
asset_probabilities = model.predict_proba(asset_history[features])[:, 1]
asset_history["failure_score"] = asset_probabilities
asset_history["alert_threshold"] = threshold
latest_reading = asset_history.iloc[-1]

sample = pd.DataFrame([{
    "temperature_c": temp,
    "vibration_mm_s": vibration,
    "rpm": rpm,
    "pressure_bar": pressure,
    "operating_hours": hours,
}])[features]
probability = float(model.predict_proba(sample)[0, 1])

def render_metric_card(column, label, value):
    column.markdown(
        f'''<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:14px;padding:1rem 1.05rem;min-height:100px">
          <div style="color:#1F2937;font-size:.88rem;font-weight:700;margin-bottom:.5rem">{label}</div>
          <div style="color:#111827;font-size:1.9rem;font-weight:700;line-height:1.2">{value}</div>
        </div>''',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
      <h1>Manutenção preditiva</h1>
      <p class="hero-sub">Visão demonstrativa da condição dos equipamentos e do desempenho do modelo.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Visão geral da base</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-note">Base demonstrativa · equipamento selecionado: <b>{selected_asset}</b>.</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
render_metric_card(k1, "Equipamentos", f"{len(asset_ids)}")
render_metric_card(k2, "Leituras na base", f"{len(df):,}")
render_metric_card(k3, "Falhas simuladas", f"{int(df.failure.sum()):,}")
render_metric_card(k4, "Recall no teste", f"{metrics['recall_failure']:.1%}")

st.markdown(f'<div class="section-title">Última leitura · {selected_asset}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-note">{latest_reading["timestamp"].strftime("%d/%m/%Y %H:%M UTC")} · resumo da leitura mais recente deste equipamento.</div>', unsafe_allow_html=True)
r1, r2, r3, r4 = st.columns(4)
render_metric_card(r1, "Temperatura", f"{latest_reading['temperature_c']:.1f} °C")
render_metric_card(r2, "Vibração", f"{latest_reading['vibration_mm_s']:.2f} mm/s")
render_metric_card(r3, "Pressão", f"{latest_reading['pressure_bar']:.1f} bar")
render_metric_card(r4, "Score estimado", f"{latest_reading['failure_score']:.1%}")

st.markdown("<br>", unsafe_allow_html=True)
tab_overview, tab_history, tab_simulator, tab_data = st.tabs(["Desempenho", "Histórico", "Simulador", "Dados e método"])

with tab_history:
    st.markdown(f'<div class="section-title">Histórico · {selected_asset}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Até 120 leituras recentes. As curvas são sintéticas e servem para demonstrar a navegação temporal.</div>', unsafe_allow_html=True)
    chart1, chart2 = st.columns(2)
    with chart1:
        st.markdown("#### Temperatura")
        st.line_chart(asset_history.set_index("timestamp")[["temperature_c"]], height=260)
    with chart2:
        st.markdown("#### Vibração")
        st.line_chart(asset_history.set_index("timestamp")[["vibration_mm_s"]], height=260)
    st.markdown("#### Score estimado ao longo do tempo")
    st.line_chart(
        asset_history.set_index("timestamp")[["failure_score", "alert_threshold"]],
        height=280,
    )
    st.caption("A linha de limiar serve como referência visual; scores sintéticos não devem orientar manutenção real.")

with tab_overview:
    st.markdown('<div class="section-title">Desempenho do modelo</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Resultados em um conjunto de teste separado da etapa usada para escolher o limiar.</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    render_metric_card(a, "ROC-AUC", f"{metrics['roc_auc']:.3f}")
    render_metric_card(b, "Average Precision", f"{metrics['average_precision']:.3f}")
    render_metric_card(c, "Precisão · falhas", f"{metrics['precision_failure']:.1%}")
    render_metric_card(d, "F1-score · falhas", f"{metrics['f1_failure']:.1%}")

    st.markdown("<br>", unsafe_allow_html=True)
    x1, x2, x3 = st.columns(3)
    render_metric_card(x1, "Falhas detectadas", metrics["true_positives"])
    render_metric_card(x2, "Falhas não detectadas", metrics["false_negatives"])
    render_metric_card(x3, "Alertas falsos", metrics["false_positives"])
    st.markdown(
        f"""
        <div class="notice-info">
          <span class="notice-info-icon">ℹ️</span>
          <span class="notice-info-text"><strong>Limiar de alerta: {threshold:.1%}.</strong> Foi escolhido na validação para buscar recall de {metrics['target_recall']:.0%}; no teste, o recall observado foi de {metrics['recall_failure']:.1%}. A meta é apenas demonstrativa e pode gerar muitos alertas falsos.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image((ROOT / "reports" / "confusion_matrix.png").read_bytes(), caption="Matriz de confusão · conjunto de teste", use_container_width=True)
    with col2:
        st.image((ROOT / "reports" / "feature_importance.png").read_bytes(), caption="Importância das variáveis no modelo", use_container_width=True)

with tab_simulator:
    st.markdown('<div class="section-title">Estimativa para a condição selecionada</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Os controles ficam na barra lateral. O resultado é uma pontuação do modelo, não um diagnóstico.</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        st.metric("Score estimado", f"{probability:.1%}")
        st.metric("Limiar atual", f"{threshold:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
    with right:
        if probability < threshold:
            if probability >= threshold * 0.5:
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
    st.markdown("#### Leituras recentes do equipamento")
    visible_columns = ["timestamp", "asset_id", *features, "failure"]
    st.dataframe(
        asset_history[visible_columns].sort_values("timestamp", ascending=False).head(20),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown('<div class="footnote">Os dados foram gerados artificialmente. Métricas e scores servem apenas para demonstração e não devem orientar manutenção real.</div>', unsafe_allow_html=True)
