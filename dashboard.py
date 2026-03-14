import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
import requests
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropSight CornBelt",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent
DATA_INTERIM = ROOT / "data" / "interim"
DATA_RAW     = ROOT / "data" / "raw"
MODELS_DIR   = ROOT / "models"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
      background: #f8f9fa; border-radius: 8px;
      padding: 16px 20px; margin: 4px 0;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #2d6a4f; }
  .metric-label { font-size: 0.85rem; color: #6c757d; margin-top: 4px; }
  .section-header {
      font-size: 1.1rem; font-weight: 600;
      color: #1a1a2e; border-bottom: 2px solid #2d6a4f;
      padding-bottom: 6px; margin-bottom: 16px;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
      background: #f0f2f6; border-radius: 6px 6px 0 0;
      padding: 8px 20px; font-weight: 500;
  }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    fm    = pd.read_parquet(DATA_INTERIM / "feature_matrix.parquet")
    preds = pd.read_parquet(DATA_INTERIM / "predictions_full.parquet")
    nass  = pd.read_csv(
        DATA_RAW / "nass" / "corn_yield_county_2000_2023.csv",
        dtype={"fips": str}
    )
    nass  = nass[nass["year"] <= 2023]
    ndvi  = pd.read_parquet(DATA_RAW / "modis" / "ndvi_county_2000_2023.parquet")
    ndvi["date"] = pd.to_datetime(ndvi["date"])
    ndvi["doy"]  = ndvi["date"].dt.dayofyear
    ndvi["year"] = ndvi["date"].dt.year
    return fm, preds, nass, ndvi

@st.cache_data
def load_county_geo():
    """Download US county boundaries from Census TIGER (cached)."""
    url = (
        "https://raw.githubusercontent.com/plotly/datasets/master/"
        "geojson-counties-fips.json"
    )
    try:
        r = requests.get(url, timeout=30)
        return r.json()
    except Exception:
        return None

fm, preds, nass, ndvi = load_data()
counties_geojson = load_county_geo()

# Merge predictions into feature matrix
fm = fm.merge(
    preds[["fips","year","pred_xgb"]],
    on=["fips","year"], how="left"
)
fm["pred_error"]   = fm["pred_xgb"] - fm["yield_bu_acre"]
fm["yield_anomaly"]= fm["yield_bu_acre"] - fm.groupby("year")["yield_bu_acre"].transform("mean")

CORN_BELT_FIPS = fm["fips"].unique().tolist()
ALL_YEARS      = sorted(fm["year"].dropna().unique().astype(int).tolist())
STATES         = ["IA","IL","IN"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Corn_leaf.svg/120px-Corn_leaf.svg.png",
             width=60)
    st.title("CropSight CornBelt")
    st.caption("US Corn Belt Yield Forecasting  |  MODIS + ERA5 + ML")
    st.divider()

    selected_tab = st.radio(
        "Navigate",
        ["🗺 Yield Map", "📈 Season View", "🔍 Explainability", "⏪ Hindcast"],
        label_visibility="collapsed"
    )
    st.divider()

    selected_year  = st.select_slider("Year", options=ALL_YEARS, value=2023)
    selected_state = st.selectbox("State", ["All"] + STATES)
    st.divider()
    st.markdown("**Model:** XGBoost baseline")
    st.markdown("**RMSE:** 12.5 bu/acre (test 2023)")
    st.markdown("**R²:** 0.432 (test 2023)")
    st.divider()
    st.caption("Data: USDA NASS · MODIS MOD13Q1 · ERA5")

# ── Filter data ───────────────────────────────────────────────────────────────
df_year = fm[fm["year"] == selected_year].copy()
if selected_state != "All":
    df_year = df_year[df_year["state"] == selected_state]

# ── Helper ────────────────────────────────────────────────────────────────────
def fmt_metric(value, suffix=""):
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}{suffix}"

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — YIELD MAP
# ═══════════════════════════════════════════════════════════════════════════════
if selected_tab == "🗺 Yield Map":
    st.markdown(f"## 🌽 Corn Yield Map — {selected_year}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Actual Yield",
                  f"{df_year['yield_bu_acre'].mean():.1f} bu/ac",
                  delta=f"{df_year['yield_bu_acre'].mean() - fm[fm['year']==selected_year-1]['yield_bu_acre'].mean():.1f} vs prior yr"
                  if selected_year > ALL_YEARS[0] else None)
    with col2:
        if "pred_xgb" in df_year.columns and df_year["pred_xgb"].notna().any():
            st.metric("Mean Predicted Yield",
                      f"{df_year['pred_xgb'].mean():.1f} bu/ac")
        else:
            st.metric("Counties", f"{df_year['fips'].nunique()}")
    with col3:
        st.metric("Best County",
                  f"{df_year.loc[df_year['yield_bu_acre'].idxmax(),'county_name']}")
    with col4:
        rmse_yr = np.sqrt(((df_year["pred_xgb"] - df_year["yield_bu_acre"])**2).mean())                   if "pred_xgb" in df_year.columns else None
        if rmse_yr:
            st.metric("RMSE this year", f"{rmse_yr:.1f} bu/ac")

    st.divider()

    map_metric = st.radio(
        "Map variable",
        ["Actual yield", "Predicted yield", "Prediction error", "Yield anomaly"],
        horizontal=True
    )

    col_map = {
        "Actual yield"    : ("yield_bu_acre", "RdYlGn",  "bu/acre"),
        "Predicted yield" : ("pred_xgb",      "RdYlGn",  "bu/acre"),
        "Prediction error": ("pred_error",    "RdBu",    "bu/acre"),
        "Yield anomaly"   : ("yield_anomaly", "RdYlGn",  "bu/acre"),
    }
    var, cmap, unit = col_map[map_metric]

    if counties_geojson and var in df_year.columns:
        fig_map = px.choropleth(
            df_year.dropna(subset=[var]),
            geojson=counties_geojson,
            locations="fips",
            color=var,
            color_continuous_scale=cmap,
            range_color=[df_year[var].quantile(0.05), df_year[var].quantile(0.95)],
            scope="usa",
            labels={var: unit},
            hover_data={"fips": False, "county_name": True,
                        "state": True, var: True,
                        "yield_bu_acre": ":.1f"},
            fitbounds="locations",
        )
        fig_map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(title=unit, thickness=12, len=0.6),
            height=480,
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("County boundary GeoJSON not loaded — check internet connection")
        # Fallback: bar chart
        top20 = df_year.nlargest(20, "yield_bu_acre")
        fig_bar = px.bar(top20, x="county_name", y="yield_bu_acre",
                         color="state", title=f"Top 20 counties by yield — {selected_year}")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Actual vs predicted scatter
    if "pred_xgb" in df_year.columns and df_year["pred_xgb"].notna().sum() > 5:
        st.markdown("#### Predicted vs actual — county level")
        fig_sc = px.scatter(
            df_year.dropna(subset=["pred_xgb"]),
            x="yield_bu_acre", y="pred_xgb",
            color="state",
            hover_data=["county_name", "year"],
            labels={"yield_bu_acre": "Actual (bu/acre)", "pred_xgb": "Predicted (bu/acre)"},
            opacity=0.7,
        )
        lo = df_year[["yield_bu_acre","pred_xgb"]].min().min() - 5
        hi = df_year[["yield_bu_acre","pred_xgb"]].max().max() + 5
        fig_sc.add_shape(type="line", x0=lo, x1=hi, y0=lo, y1=hi,
                         line=dict(dash="dash", color="gray", width=1.5))
        fig_sc.update_layout(height=380)
        st.plotly_chart(fig_sc, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEASON VIEW
# ═══════════════════════════════════════════════════════════════════════════════
elif selected_tab == "📈 Season View":
    st.markdown("## 📈 Season Trajectory")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        county_options = sorted(
            fm[fm["state"] == (selected_state if selected_state != "All" else "IA")]
            ["county_name"].unique().tolist()
        )
        selected_county = st.selectbox("County", county_options)
        compare_year    = st.selectbox(
            "Compare year", [None] + ALL_YEARS,
            format_func=lambda x: "None" if x is None else str(x)
        )

    fips_sel = fm[fm["county_name"] == selected_county]["fips"].iloc[0]                if len(fm[fm["county_name"] == selected_county]) > 0 else None

    with col_b:
        if fips_sel:
            ndvi_county = ndvi[
                (ndvi["fips"] == fips_sel) &
                (ndvi["doy"] >= 60) & (ndvi["doy"] <= 330)
            ]

            years_to_plot = [selected_year]
            if compare_year and compare_year != selected_year:
                years_to_plot.append(compare_year)

            fig_ndvi = go.Figure()
            colors   = ["#2d6a4f", "#e76f51", "#457b9d", "#e9c46a"]

            for i, yr in enumerate(years_to_plot):
                sub = ndvi_county[ndvi_county["year"] == yr].sort_values("doy")
                if sub.empty:
                    continue
                actual_yield = nass[
                    (nass["fips"] == fips_sel) & (nass["year"] == yr)
                ]["yield_bu_acre"].values
                lbl = f"{yr} (yield: {actual_yield[0]:.0f} bu/ac)"                       if len(actual_yield) > 0 else str(yr)

                fig_ndvi.add_trace(go.Scatter(
                    x=sub["doy"], y=sub["ndvi_mean"],
                    mode="lines+markers", name=lbl,
                    line=dict(color=colors[i], width=2.5),
                    marker=dict(size=5),
                ))

            # Growing season phase bands
            for s, e, lbl, clr in [
                (110, 150, "Planting",   "rgba(255,235,59,0.12)"),
                (150, 180, "Vegetative", "rgba(76,175,80,0.12)"),
                (180, 220, "Silking",    "rgba(244,67,54,0.15)"),
                (220, 270, "Grain fill", "rgba(33,150,243,0.12)"),
            ]:
                fig_ndvi.add_vrect(x0=s, x1=e, fillcolor=clr, layer="below",
                                   line_width=0, annotation_text=lbl,
                                   annotation_position="top left",
                                   annotation_font_size=10)

            fig_ndvi.update_layout(
                title=f"NDVI seasonal trajectory — {selected_county}",
                xaxis_title="Day of year",
                yaxis_title="NDVI",
                xaxis=dict(range=[60, 330]),
                height=400, legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_ndvi, use_container_width=True)

    # Historical yield trend for selected county
    if fips_sel:
        st.markdown("#### Historical yield trend")
        hist = nass[nass["fips"] == fips_sel].sort_values("year")
        hist_pred = fm[fm["fips"] == fips_sel].sort_values("year")

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist["year"], y=hist["yield_bu_acre"],
            mode="lines+markers", name="Actual",
            line=dict(color="#2d6a4f", width=2.5),
        ))
        if "pred_xgb" in hist_pred.columns:
            fig_hist.add_trace(go.Scatter(
                x=hist_pred["year"], y=hist_pred["pred_xgb"],
                mode="lines", name="XGBoost prediction",
                line=dict(color="#e76f51", width=2, dash="dash"),
            ))
        fig_hist.add_vline(x=2012, line_dash="dot", line_color="red",
                           annotation_text="2012 drought", annotation_position="top right")
        fig_hist.update_layout(
            xaxis_title="Year", yaxis_title="Yield (bu/acre)",
            height=320, legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # State-level yield comparison
    st.markdown("#### State mean yield — all years")
    state_annual = nass.groupby(["state","year"])["yield_bu_acre"].mean().reset_index()
    fig_state = px.line(
        state_annual, x="year", y="yield_bu_acre",
        color="state", markers=True,
        labels={"yield_bu_acre": "Mean yield (bu/acre)", "year": "Year"},
        color_discrete_sequence=["#2d6a4f","#457b9d","#e76f51"],
    )
    fig_state.add_vline(x=2012, line_dash="dot", line_color="red",
                        annotation_text="2012 drought")
    fig_state.update_layout(height=320)
    st.plotly_chart(fig_state, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif selected_tab == "🔍 Explainability":
    st.markdown("## 🔍 Model Explainability")

    # Feature correlations with yield
    st.markdown("#### Feature correlation with corn yield (train set)")

    FEATURE_COLS = [c for c in fm.columns
                    if c not in ["fips","state","county_name","year",
                                 "yield_bu_acre","pred_xgb","pred_error","yield_anomaly"]]
    train_fm = fm[fm["year"] <= 2021]
    corr = (train_fm[FEATURE_COLS + ["yield_bu_acre"]]
            .corr()["yield_bu_acre"]
            .drop("yield_bu_acre")
            .sort_values(key=abs, ascending=False)
            .head(20))

    fig_corr = go.Figure(go.Bar(
        x=corr.values,
        y=corr.index,
        orientation="h",
        marker_color=["#2d6a4f" if v > 0 else "#e76f51" for v in corr.values],
        text=[f"{v:+.3f}" for v in corr.values],
        textposition="outside",
    ))
    fig_corr.update_layout(
        title="Pearson correlation with yield (top 20 features)",
        xaxis_title="Correlation",
        yaxis=dict(autorange="reversed"),
        height=550,
        xaxis=dict(range=[-0.75, 0.75]),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # County-year feature profile
    st.markdown("#### Feature profile — select a county-year")
    col1, col2 = st.columns(2)
    with col1:
        exp_county = st.selectbox("County", sorted(fm["county_name"].unique()), key="exp_c")
    with col2:
        exp_year   = st.selectbox("Year", ALL_YEARS, index=len(ALL_YEARS)-1, key="exp_y")

    row = fm[(fm["county_name"] == exp_county) & (fm["year"] == exp_year)]
    if not row.empty:
        row = row.iloc[0]
        actual = row["yield_bu_acre"]
        pred   = row.get("pred_xgb", np.nan)

        c1, c2, c3 = st.columns(3)
        c1.metric("Actual yield",    f"{actual:.1f} bu/ac")
        c2.metric("Predicted yield", f"{pred:.1f} bu/ac" if not pd.isna(pred) else "N/A")
        c3.metric("Error",
                  f"{pred - actual:+.1f} bu/ac" if not pd.isna(pred) else "N/A",
                  delta_color="inverse")

        # Show top features for this row vs Corn Belt mean
        feat_vals = row[FEATURE_COLS]
        cb_means  = fm[FEATURE_COLS].mean()
        cb_stds   = fm[FEATURE_COLS].std().replace(0, 1)
        z_scores  = ((feat_vals - cb_means) / cb_stds).sort_values(key=abs, ascending=False).head(12)

        fig_z = go.Figure(go.Bar(
            x=z_scores.values,
            y=z_scores.index,
            orientation="h",
            marker_color=["#2d6a4f" if v > 0 else "#e76f51" for v in z_scores.values],
            text=[f"{v:+.2f}σ" for v in z_scores.values],
            textposition="outside",
        ))
        fig_z.update_layout(
            title=f"Feature z-scores vs Corn Belt mean — {exp_county} {exp_year}",
            xaxis_title="Standard deviations from mean",
            yaxis=dict(autorange="reversed"),
            height=420,
        )
        st.plotly_chart(fig_z, use_container_width=True)
        st.caption("Positive = above average | Negative = below average | Values in σ units")

    # Physics constraint validation
    st.markdown("#### Physics constraint validation")
    st.markdown("Verifying the PINN constraint: higher `water_stress_frac` → higher predicted yield")
    if "pred_xgb" in fm.columns and "water_stress_frac" in fm.columns:
        sample = fm.dropna(subset=["pred_xgb","water_stress_frac"]).sample(
            min(500, len(fm)), random_state=42)
        fig_phys = px.scatter(
            sample, x="water_stress_frac", y="pred_xgb",
            color="year", opacity=0.5,
            labels={"water_stress_frac": "Water stress fraction (higher=less stress)",
                    "pred_xgb": "Predicted yield (bu/acre)"},
            title="Water stress fraction vs predicted yield (should trend upward)",
            trendline="ols",
        )
        fig_phys.update_layout(height=360)
        st.plotly_chart(fig_phys, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HINDCAST
# ═══════════════════════════════════════════════════════════════════════════════
elif selected_tab == "⏪ Hindcast":
    st.markdown("## ⏪ Hindcast — Year Replay")
    st.markdown("Replay any historical year to see how predictions would have evolved.")

    col1, col2 = st.columns([1, 2])
    with col1:
        hindcast_year = st.selectbox("Hindcast year", ALL_YEARS,
                                     index=ALL_YEARS.index(2012))
        hindcast_state = st.selectbox("State", ["All"] + STATES, key="hc_state")

    hc = fm[fm["year"] == hindcast_year].copy()
    if hindcast_state != "All":
        hc = hc[hc["state"] == hindcast_state]

    with col2:
        if "pred_xgb" in hc.columns:
            rmse_hc = np.sqrt(((hc["pred_xgb"] - hc["yield_bu_acre"])**2).mean())
            bias_hc = (hc["pred_xgb"] - hc["yield_bu_acre"]).mean()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Year",          str(hindcast_year))
            c2.metric("Actual mean",   f"{hc['yield_bu_acre'].mean():.1f} bu/ac")
            c3.metric("RMSE",          f"{rmse_hc:.1f} bu/ac")
            c4.metric("Bias",          f"{bias_hc:+.1f} bu/ac")

    st.divider()

    # Choropleth for hindcast year
    if counties_geojson and "pred_xgb" in hc.columns:
        hc_metric = st.radio("Map variable",
                             ["Actual yield","Predicted yield","Prediction error"],
                             horizontal=True, key="hc_metric")
        hc_col = {"Actual yield":"yield_bu_acre",
                  "Predicted yield":"pred_xgb",
                  "Prediction error":"pred_error"}[hc_metric]
        hc["pred_error"] = hc["pred_xgb"] - hc["yield_bu_acre"]

        fig_hc = px.choropleth(
            hc.dropna(subset=[hc_col]),
            geojson=counties_geojson,
            locations="fips",
            color=hc_col,
            color_continuous_scale="RdYlGn" if "error" not in hc_col else "RdBu",
            scope="usa",
            fitbounds="locations",
            hover_data={"county_name": True, "state": True,
                        "yield_bu_acre": ":.1f", hc_col: ":.1f"},
            labels={hc_col: "bu/acre"},
        )
        fig_hc.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=420)
        st.plotly_chart(fig_hc, use_container_width=True)

    # Year-over-year RMSE trend
    st.markdown("#### Model RMSE by year (hindcast accuracy over time)")
    yearly_rmse = []
    for yr in ALL_YEARS:
        sub = fm[fm["year"] == yr]
        if "pred_xgb" in sub.columns and sub["pred_xgb"].notna().sum() > 10:
            rmse = np.sqrt(((sub["pred_xgb"] - sub["yield_bu_acre"])**2).mean())
            yearly_rmse.append({"year": yr, "rmse": rmse,
                                 "n_counties": sub["fips"].nunique()})

    if yearly_rmse:
        rmse_df = pd.DataFrame(yearly_rmse)
        fig_rmse = go.Figure()
        fig_rmse.add_trace(go.Bar(
            x=rmse_df["year"], y=rmse_df["rmse"],
            marker_color=["#e76f51" if yr == 2012 else "#2d6a4f"
                          for yr in rmse_df["year"]],
            name="RMSE",
        ))
        fig_rmse.add_hline(y=rmse_df["rmse"].mean(), line_dash="dash",
                           line_color="gray",
                           annotation_text=f"Mean {rmse_df['rmse'].mean():.1f}",
                           annotation_position="right")
        fig_rmse.update_layout(
            xaxis_title="Year",
            yaxis_title="RMSE (bu/acre)",
            title="XGBoost hindcast RMSE by year (red = 2012 drought)",
            height=320,
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    # Drought year deep-dive
    if hindcast_year in [2012, 2002, 2011, 2019]:
        st.markdown(f"#### {hindcast_year} anomaly — how far below normal?")
        base = fm[fm["year"].isin(range(hindcast_year-5, hindcast_year))]               .groupby("fips")["yield_bu_acre"].mean()
        hc2  = hc.set_index("fips")["yield_bu_acre"]
        anom = (hc2 - base).dropna()
        fig_anom = px.histogram(
            anom, nbins=30,
            labels={"value": "Yield anomaly vs 5-yr prior mean (bu/acre)"},
            title=f"{hindcast_year} county yield anomalies vs prior 5-year average",
            color_discrete_sequence=["#e76f51"],
        )
        fig_anom.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_anom.update_layout(height=300)
        st.plotly_chart(fig_anom, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "CropSight CornBelt · M.S. Atmospheric Science, Texas A&M  |  "
    "Data: USDA NASS · NASA MODIS MOD13Q1 · ERA5 Reanalysis  |  "
    "Model: XGBoost + LSTM + PINN ensemble"
)
