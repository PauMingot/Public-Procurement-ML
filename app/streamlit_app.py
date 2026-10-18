"""Interactive Public Procurement Auction Simulator (Streamlit App).

Simulates competitor bids and predicts award prices for Spanish public tenders
using trained Machine Learning pipelines (XGBoost) and contractor profiles.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from config import DATA_DIR, OUTPUT_DIR

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Public Procurement ML Simulator",
    page_icon="🏛️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# File Path Resolution
# ---------------------------------------------------------------------------
POSSIBLE_PROFILE_PATHS = [
    DATA_DIR / "perfiles_empresas.parquet",
    PROJECT_ROOT / "data" / "perfiles_empresas.parquet",
    PROJECT_ROOT / "perfiles_empresas.parquet",
    Path(__file__).parent / "perfiles_empresas.parquet",
]

POSSIBLE_MODEL_PATHS = [
    OUTPUT_DIR / "xgboost_bid" / "xgb_bid_winner.joblib",
    PROJECT_ROOT / "outputs" / "xgboost_bid" / "xgb_bid_winner.joblib",
    PROJECT_ROOT / "xgb_ganador_baja_comprimido.pkl",
    Path(__file__).parent / "xgb_ganador_baja_comprimido.pkl",
]


def resolve_first_existing(paths: list[Path]) -> Path | None:
    """Return first path that exists on disk."""
    for p in paths:
        if p.exists():
            return p
    return None


@st.cache_resource
def load_data_and_model() -> tuple[pd.DataFrame, object]:
    """Load contractor profile database and fitted ML model pipeline."""
    profiles_path = resolve_first_existing(POSSIBLE_PROFILE_PATHS)
    model_path = resolve_first_existing(POSSIBLE_MODEL_PATHS)

    if profiles_path is None or model_path is None:
        raise FileNotFoundError(
            f"Required assets missing.\n"
            f"- Profiles path: {profiles_path or 'Not found'}\n"
            f"- Model path: {model_path or 'Not found'}\n"
            f"Please run 'python scripts/build_company_profiles.py' and "
            f"'python models/xgboost/xgb_bid.py' to generate artifacts."
        )

    df_profiles = pd.read_parquet(profiles_path)
    model = joblib.load(model_path)
    return df_profiles, model


def pct_bid_to_euros(pct_bid: np.ndarray | float, budget: float) -> np.ndarray:
    """Convert predicted percentage discount (0-100) to euro award price."""
    safe_pct = np.clip(np.asarray(pct_bid, dtype=float), 0.0, 100.0)
    return float(budget) * (1.0 - safe_pct / 100.0)


def find_rivals(
    tender_dict: dict,
    df_profiles: pd.DataFrame,
    tolerance: float,
    type_threshold: float,
    geo_threshold: float,
    filter_geo: bool,
    max_inactivity_days: int | None,
) -> pd.DataFrame:
    """Filter candidate contractors based on specialization, size, and location."""
    rivals = df_profiles.copy()

    # 1. Contract Type Specialization
    contract_type_norm = tender_dict["tipo_contrato"].lower().replace(" ", "_")
    col_type = f"pct_hist_{contract_type_norm}"
    if col_type in rivals.columns:
        rivals = rivals[rivals[col_type] >= type_threshold]

    # 2. Contractor Size Volume Match
    budget = tender_dict["lote_presupuesto_base_sin_impuestos"]
    if "presupuesto_medio_hist" in rivals.columns:
        min_p = budget * (1.0 - tolerance)
        max_p = budget * (1.0 + tolerance)
        rivals = rivals[rivals["presupuesto_medio_hist"].between(min_p, max_p)]

    # 3. Geographic Experience Match
    if filter_geo:
        region_norm = tender_dict["com_aut_licitacion"].lower().replace(" ", "_")
        col_geo = f"pct_hist_com_aut_{region_norm}"
        if col_geo in rivals.columns:
            rivals = rivals[rivals[col_geo] >= geo_threshold]

    # 4. Inactivity Recency Threshold
    if max_inactivity_days is not None and "fecha_ultima_licitacion_empresa" in rivals.columns:
        ref_date = pd.to_datetime(tender_dict["fecha_primera_publicacion"])
        days = (ref_date - pd.to_datetime(rivals["fecha_ultima_licitacion_empresa"])).dt.days
        rivals = rivals[days <= max_inactivity_days]

    if len(rivals) > 100 and "n_licitaciones_hist" in rivals.columns:
        rivals = rivals.sort_values("n_licitaciones_hist", ascending=False).head(100)

    return rivals


def simulate_auction(tender_dict: dict, df_rivals: pd.DataFrame, model: object) -> pd.DataFrame | None:
    """Predict bid offers for each candidate contractor."""
    sim_df = pd.DataFrame([tender_dict] * len(df_rivals))

    for col in df_rivals.columns:
        if col != "fecha_ultima_licitacion_empresa":
            sim_df[col] = df_rivals[col].values

    if "fecha_ultima_licitacion_empresa" in df_rivals.columns:
        pub_date = pd.to_datetime(tender_dict["fecha_primera_publicacion"])
        last_dates = pd.to_datetime(df_rivals["fecha_ultima_licitacion_empresa"].values)
        sim_df["dias_desde_ultima_licitacion"] = (pub_date - last_dates).days
        sim_df["dias_desde_ultima_licitacion"] = sim_df["dias_desde_ultima_licitacion"].fillna(365).clip(lower=0)

    sim_df["presupuesto_base_sin_impuestos"] = sim_df["lote_presupuesto_base_sin_impuestos"]
    if "lote" not in sim_df.columns:
        sim_df["lote"] = 1
    if "peso_relativo_lote" not in sim_df.columns:
        sim_df["peso_relativo_lote"] = 1.0
    for c in ["es_loteado", "es_anomalia_temporal"]:
        sim_df[c] = 0

    cat_cols = sim_df.select_dtypes(include=["object", "category"]).columns
    for c in cat_cols:
        sim_df[c] = sim_df[c].astype(str)

    try:
        pred_pct = model.predict(sim_df)
        budget = float(tender_dict["lote_presupuesto_base_sin_impuestos"])
        pred_euros = pct_bid_to_euros(pred_pct, budget)

        df_res = df_rivals.copy()
        df_res["predicted_bid_eur"] = pred_euros
        df_res["estimated_discount_pct"] = (budget - pred_euros) / budget * 100.0

        if "dias_desde_ultima_licitacion" in sim_df.columns:
            df_res["days_since_last_bid"] = sim_df["dias_desde_ultima_licitacion"].values

        return df_res.sort_values("predicted_bid_eur")
    except Exception as err:
        st.error(f"Inference error: {err}")
        return None


# ---------------------------------------------------------------------------
# Streamlit User Interface
# ---------------------------------------------------------------------------
st.title("🏛️ Public Procurement Auction Simulator")
st.markdown(
    "Predict contractor bid discounts and expected award prices for Spanish public tenders "
    "using Machine Learning models trained on official PLACE open procurement data."
)

try:
    df_profiles, model = load_data_and_model()
    st.sidebar.success(f"System Online: {len(df_profiles):,} contractor profiles loaded.")
except Exception as err:
    st.warning(
        f"⚠️ System in preview mode: {err}\n\n"
        "To run live simulations, generate company profiles and model artifacts using the instructions in README.md."
    )
    df_profiles, model = None, None

if df_profiles is not None and model is not None:
    all_cols = df_profiles.columns.tolist()
    types_list = sorted(list(set(c.replace("pct_hist_", "") for c in all_cols if c.startswith("pct_hist_") and "com_aut" not in c)))
    regions_list = sorted(list(set(c.replace("pct_hist_com_aut_", "") for c in all_cols if c.startswith("pct_hist_com_aut_"))))

    with st.sidebar:
        st.header("1. Tender Specification")
        budget = st.number_input("Base Budget (€ without tax)", value=150000, step=5000)
        contract_type = st.selectbox("Contract Type", types_list)
        region = st.selectbox("Autonomous Community", regions_list)

        st.header("2. Competitor Filtering")
        tolerance = st.slider("Budget Size Range (±%)", 0.1, 1.0, 0.6)
        type_threshold = st.slider(f"Min. Specialization in {contract_type}", 0.0, 1.0, 0.2)
        filter_geo = st.checkbox("Filter by Regional Experience", value=True)
        geo_threshold = 0.0
        if filter_geo:
            geo_threshold = st.slider(f"Min. Experience in {region}", 0.0, 1.0, 0.1)
        max_inactivity_days = st.number_input("Max. Inactivity (Days)", value=730)
        btn_calc = st.button("🔍 Simulate Auction", type="primary")

    if btn_calc:
        new_tender = {
            "lote_presupuesto_base_sin_impuestos": budget,
            "valor_estimado_imputado": budget,
            "duracion_proceso_dias": 45,
            "anio": 2025,
            "mes": 6,
            "tipo_contrato": contract_type,
            "com_aut_licitacion": region,
            "sistema_contratacion": "No aplica",
            "tipo_procedimiento": "Abierto",
            "tipo_administracion": "Administracion Local",
            "lote_pyme": "No",
            "lote_tipo_id_adjudicatario": "NIF Definitivo",
            "organo_cluster_label": "ORG_Ayto_General",
            "objeto_cluster_label": "OBJ_General",
            "fecha_primera_publicacion": "2025-06-01",
        }

        rivals = find_rivals(new_tender, df_profiles, tolerance, type_threshold, geo_threshold, filter_geo, max_inactivity_days)

        if not rivals.empty:
            results = simulate_auction(new_tender, rivals, model)
            if results is not None:
                median_bid = float(results["predicted_bid_eur"].median())
                min_bid = float(results["predicted_bid_eur"].min())
                median_discount = float(results["estimated_discount_pct"].median())

                col1, col2, col3 = st.columns(3)
                col1.metric("Tender Base Budget", f"{budget:,.0f} €")
                col2.metric("Predicted Market Price (Median)", f"{median_bid:,.0f} €", delta=f"-{median_discount:.1f}% discount")
                col3.metric("Most Aggressive Bid", f"{min_bid:,.0f} €")

                st.subheader("Distribution of Predicted Competitor Bids")
                fig, ax = plt.subplots(figsize=(10, 3.5))
                sns.histplot(results["predicted_bid_eur"], kde=True, ax=ax, color="#4C72B0")
                ax.axvline(median_bid, color="red", linestyle="--", label=f"Median Bid ({median_bid:,.0f} €)")
                ax.set_xlabel("Bid Amount (€)")
                ax.legend()
                st.pyplot(fig)

                st.subheader(f"Ranked Competitor Predictions ({len(rivals)} candidates)")
                cols_display = ["predicted_bid_eur", "estimated_discount_pct", "descuento_medio_hist", "n_licitaciones_hist"]
                st.dataframe(
                    results[cols_display].style.format({
                        "predicted_bid_eur": "{:,.0f} €",
                        "estimated_discount_pct": "{:.1f}%",
                        "descuento_medio_hist": "{:.1%}",
                        "n_licitaciones_hist": "{:.0f}",
                    }),
                    use_container_width=True,
                )
        else:
            st.warning("No contractors matched the specified filters. Try increasing the budget range tolerance.")
    else:
        st.info("👈 Configure the tender specification on the left sidebar and click 'Simulate Auction'.")
