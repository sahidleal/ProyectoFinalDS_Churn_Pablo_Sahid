import io
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    letter-spacing: -0.5px;
}

/* Tarjetas de métricas con relieve, no planas */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1a1f2b, #161b24);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}

div[data-testid="stMetricValue"] {
    font-family: 'Sora', sans-serif;
    font-size: 2rem;
}

div[data-testid="stMetricLabel"] {
    color: #9ca3af;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Tabla con bordes suaves en vez de la grid plana por defecto */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Botones con más presencia */
.stButton button, .stDownloadButton button {
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: transform 0.15s ease;
}
.stButton button:hover, .stDownloadButton button:hover {
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Configuración general
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Predictor de Churn", page_icon="📉", layout="wide")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"   # app en src/, modelos en models/

NUMERICAS = [
    "MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge", "DirectorAssistedCalls", "OverageMinutes",
    "RoamingCalls", "PercChangeMinutes", "PercChangeRevenues", "DroppedCalls", "BlockedCalls",
    "UnansweredCalls", "CustomerCareCalls", "ThreewayCalls", "ReceivedCalls", "OutboundCalls",
    "InboundCalls", "PeakCallsInOut", "OffPeakCallsInOut", "CallForwardingCalls", "CallWaitingCalls",
    "MonthsInService", "UniqueSubs", "ActiveSubs", "Handsets", "HandsetModels", "CurrentEquipmentDays",
    "AgeHH1", "AgeHH2", "RetentionCalls", "RetentionOffersAccepted", "ReferralsMadeBySubscriber",
    "IncomeGroup", "AdjustmentsToCreditRating",
]
# mismo orden que en el entrenamiento
CATEGORICAS = [
    "ServiceArea", "ChildrenInHH", "HandsetRefurbished", "HandsetWebCapable", "TruckOwner", "RVOwner",
    "Homeownership", "BuysViaMailOrder", "RespondsToMailOffers", "OptOutMailings", "NonUSTravel",
    "OwnsComputer", "HasCreditCard", "NewCellphoneUser", "OwnsMotorcycle", "HandsetPrice",
    "MadeCallToRetentionTeam", "CreditRating", "PrizmCode", "Occupation", "MaritalStatus",
]
PREDICTORAS = NUMERICAS + [c + "_n" for c in CATEGORICAS]
REQUERIDAS = ["CustomerID"] + NUMERICAS + CATEGORICAS

COLOR_ALTO, COLOR_MEDIO, COLOR_BAJO = "#B42318", "#C98A1B", "#2F7D5B"

# ----------------------------------------------------------------------------
# Estilo
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"], .stMarkdown, button, input { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
    .block-container { max-width: 1180px; padding-top: 3.5rem; }
    footer { visibility: hidden; }

    .titulo { font-size: 2rem; line-height: 1.3; font-weight: 600; letter-spacing: -0.02em; margin: 0; }
    .subtitulo { color: #5b6673; margin: .25rem 0 1.6rem 0; max-width: 62ch; line-height: 1.5; }

    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e3e7ec; border-radius: 6px; padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] p { color: #5b6673; font-size: .85rem; }
    [data-testid="stMetricValue"] { font-weight: 600; letter-spacing: -0.02em; }

    .barra { display: flex; height: 14px; border-radius: 3px; overflow: hidden; margin: .4rem 0 .5rem 0; }
    .leyenda { display: flex; gap: 1.4rem; font-size: .85rem; color: #3b4652; }
    .punto { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Funciones
# ----------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load(MODELS_DIR / "best_model_XGBoost.joblib")


@st.cache_data
def cargar_reglas(columna):
    with open(MODELS_DIR / f"{columna}_transformation_rules.json") as f:
        return json.load(f)


def leer_archivo(archivo):
    extension = os.path.splitext(archivo.name)[1].lower()
    if extension == ".csv":
        return pd.read_csv(archivo, dtype={"HandsetPrice": str})
    return pd.read_excel(archivo, dtype={"HandsetPrice": str})


def validar(df):
    """Devuelve una lista de problemas encontrados (vacía si el archivo es válido)."""
    problemas = []
    faltan = [c for c in REQUERIDAS if c not in df.columns]
    if faltan:
        problemas.append(f"Faltan estas columnas: {', '.join(faltan)}.")
        return problemas
    con_nulos = [c for c in REQUERIDAS if df[c].isnull().any()]
    if con_nulos:
        problemas.append(f"Hay valores vacíos en: {', '.join(con_nulos)}. Completa esos datos y vuelve a subir el archivo.")
    return problemas


def predecir(df):
    """Devuelve un DataFrame con el ID del cliente y su probabilidad de churn (0-100)."""
    modelo = cargar_modelo()
    datos = df.copy()
    datos["ServiceArea"] = datos["ServiceArea"].astype(str).str[:3]
    for col in CATEGORICAS:
        datos[col + "_n"] = datos[col].map(cargar_reglas(col))
    probas = modelo.predict_proba(datos[PREDICTORAS])
    # clase 0 = "Yes" (churn), según la factorización del notebook
    return pd.DataFrame({"Cliente": df["CustomerID"].values, "Probabilidad": probas[:, 0] * 100})


def nivel_riesgo(prob, medio_min, alto_min):
    return "Alto" if prob >= alto_min else ("Medio" if prob >= medio_min else "Bajo")


# ----------------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Cómo funciona")
    st.markdown(
        "1. Sube el archivo con los datos de tus clientes.\n"
        "2. Pulsa **Calcular predicción**.\n"
        "3. Revisa el riesgo de cada cliente y descarga el resultado."
    )
    st.divider()
    st.subheader("Niveles de riesgo")
    medio_min, alto_min = st.slider(
        "Tramo de riesgo medio (%)", 0, 100, (30, 50),
        help="Por debajo del tramo, el riesgo es bajo. Por encima, alto.",
    )
    st.caption(f"Bajo: menos de {medio_min}% · Medio: {medio_min}%–{alto_min}% · Alto: {alto_min}% o más")

# ----------------------------------------------------------------------------
# Cabecera y carga de datos
# ----------------------------------------------------------------------------
st.markdown('<p class="titulo">Predictor de Churn</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitulo">Estima qué clientes tienen más probabilidad de darse de baja, '
    "para que puedas priorizar las acciones de retención.</p>",
    unsafe_allow_html=True,
)

archivo = st.file_uploader(
    "Archivo de clientes", type=["csv", "xlsx", "xls"], help="Formatos admitidos: CSV, XLSX y XLS."
)

if archivo is None:
    st.session_state.pop("resultados", None)
    st.info("Sube un archivo para empezar.")
    st.stop()

try:
    dataframe = leer_archivo(archivo)
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

problemas = validar(dataframe)
if problemas:
    for p in problemas:
        st.error(p)
    st.stop()

st.success(f"Archivo cargado: {len(dataframe):,} clientes.")
with st.expander("Ver vista previa de los datos"):
    st.dataframe(dataframe.head(10), use_container_width=True)

# Reiniciar resultados si el usuario cambia de archivo
firma = (archivo.name, archivo.size)
if st.session_state.get("firma") != firma:
    st.session_state.pop("resultados", None)
    st.session_state["firma"] = firma

if st.button("Calcular predicción", type="primary"):
    with st.spinner("Calculando..."):
        st.session_state["resultados"] = predecir(dataframe)

if "resultados" not in st.session_state:
    st.stop()

# ----------------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------------
res = st.session_state["resultados"].copy()
res["Nivel de riesgo"] = res["Probabilidad"].apply(lambda p: nivel_riesgo(p, medio_min, alto_min))
res["Baja prevista"] = res["Probabilidad"].apply(lambda p: "Sí" if p >= 50 else "No")

total = len(res)
n_alto = int((res["Nivel de riesgo"] == "Alto").sum())
n_medio = int((res["Nivel de riesgo"] == "Medio").sum())
n_bajo = total - n_alto - n_medio

st.subheader("Resultados")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes analizados", f"{total:,}")
c2.metric("Riesgo alto", f"{n_alto:,}")
c3.metric("Porcentaje en riesgo alto", f"{n_alto / total:.1%}")
c4.metric("Probabilidad media", f"{res['Probabilidad'].mean():.1f}%")

st.markdown(
    f"""
    <div class="barra">
      <div style="width:{n_alto / total * 100}%;background:{COLOR_ALTO}"></div>
      <div style="width:{n_medio / total * 100}%;background:{COLOR_MEDIO}"></div>
      <div style="width:{n_bajo / total * 100}%;background:{COLOR_BAJO}"></div>
    </div>
    <div class="leyenda">
      <span><span class="punto" style="background:{COLOR_ALTO}"></span>Alto: {n_alto}</span>
      <span><span class="punto" style="background:{COLOR_MEDIO}"></span>Medio: {n_medio}</span>
      <span><span class="punto" style="background:{COLOR_BAJO}"></span>Bajo: {n_bajo}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

f1, f2 = st.columns([2, 1])
filtro = f1.radio("Mostrar", ["Todos", "Riesgo alto", "Riesgo alto y medio"], horizontal=True)
busqueda = f2.text_input("Buscar cliente", placeholder="ID de cliente")

vista = res
if filtro == "Riesgo alto":
    vista = vista[vista["Nivel de riesgo"] == "Alto"]
elif filtro == "Riesgo alto y medio":
    vista = vista[vista["Nivel de riesgo"] != "Bajo"]
if busqueda.strip():
    vista = vista[vista["Cliente"].astype(str).str.contains(busqueda.strip())]
vista = vista.sort_values("Probabilidad", ascending=False).reset_index(drop=True)
vista = vista[["Cliente", "Nivel de riesgo", "Probabilidad", "Baja prevista"]]

colores = {"Alto": COLOR_ALTO, "Medio": COLOR_MEDIO, "Bajo": COLOR_BAJO}
estilo = vista.style
aplicar = getattr(estilo, "map", None) or estilo.applymap   # 'map' en pandas >= 2.1
estilo = aplicar(lambda v: f"color: {colores.get(v, 'inherit')}; font-weight: 600", subset=["Nivel de riesgo"])

st.caption(f"{len(vista):,} clientes, ordenados de mayor a menor probabilidad.")
st.dataframe(
    estilo,
    use_container_width=True,
    hide_index=True,
    height=420,
    column_config={
        "Cliente": st.column_config.TextColumn("Cliente"),
        "Probabilidad": st.column_config.ProgressColumn(
            "Probabilidad de churn", format="%.1f%%", min_value=0, max_value=100
        ),
    },
)

descarga = vista.rename(columns={"Probabilidad": "Probabilidad de churn (%)"}).round(1)

d1, d2 = st.columns([1, 3])
formato = d1.selectbox("Formato de descarga", ["CSV", "Excel"], label_visibility="collapsed")

if formato == "CSV":
    datos_descarga = descarga.to_csv(index=False).encode("utf-8-sig")
    nombre_archivo = "predicciones_churn.csv"
    tipo_mime = "text/csv"
else:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        descarga.to_excel(writer, index=False, sheet_name="Predicciones")
    datos_descarga = buffer.getvalue()
    nombre_archivo = "predicciones_churn.xlsx"
    tipo_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

d2.download_button(
    f"Descargar resultados ({formato})",
    data=datos_descarga,
    file_name=nombre_archivo,
    mime=tipo_mime,
)
