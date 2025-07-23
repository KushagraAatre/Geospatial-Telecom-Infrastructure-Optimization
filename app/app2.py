import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

st.set_page_config(page_title="Telecom Tower ML & Predictive Analytics", layout="wide")
st.title("📡 Advanced Telecom Tower Analytics & ML Insights")

# --- DATA UPLOAD & CLEANING ---
uploaded = st.sidebar.file_uploader("Upload your OpenCelliD-style CSV file", type=["csv"])
if not uploaded:
    st.info("⬆️ Please upload a CSV file to get started.")
    st.stop()

# ---- Caching the Data Loading/Cleaning ----
@st.cache_data
def load_and_clean(uploaded):
    COLS_13 = ["radio", "mcc", "mnc", "area", "cell", "unit",
               "lon", "lat", "range", "samples", "changeable",
               "created", "updated"]
    COLS_14 = COLS_13 + ["average_signal"]
    COLS_15 = COLS_14 + ["unknown"]
    df = pd.read_csv(uploaded)
    if not set(["lat", "lon"]).issubset(df.columns):
        if len(df.columns) == 13:
            df.columns = COLS_13
        elif len(df.columns) == 14:
            df.columns = COLS_14
        elif len(df.columns) == 15:
            df.columns = COLS_15
        else:
            raise ValueError(f"Unexpected columns: {df.columns.tolist()}")
    df = df[(df['lat'].notnull()) & (df['lon'].notnull()) & (df['lat'] != 0) & (df['lon'] != 0)]
    df["lat_bin"] = df["lat"].round()
    df["lon_bin"] = df["lon"].round()
    df["created"] = pd.to_datetime(df["created"], errors="coerce")
    df["year"] = df["created"].dt.year
    operator_map = {
        310260: "T-Mobile US",
        310410: "AT&T US",
        311480: "Verizon US",
        310120: "Sprint"
    }
    df["operator"] = (df["mcc"].astype(str) + df["mnc"].astype(str)).astype(int)
    df["operator_name"] = df["operator"].map(operator_map).fillna("Other")
    return df

try:
    df = load_and_clean(uploaded)
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# --- Clustering (KMeans) Caching ---
@st.cache_resource
def kmeans_fit(coords, n_clusters):
    return KMeans(n_clusters=n_clusters, random_state=42).fit(coords)

# --- 1. PREDICTIVE NEW TOWER LOCATIONS (OVERALL) ---
st.header("📍 Predicting Optimal New Tower Locations (All Aggregators)")
st.markdown("We use K-Means clustering to identify underserved regions (potential spots for new towers) based on the geographic distribution of all existing towers.")

n_clusters = st.slider("Number of Predicted New Towers (Clusters)", min_value=3, max_value=25, value=10)
coords = df[['lat', 'lon']].dropna()
if len(coords) >= n_clusters:
    kmeans = kmeans_fit(coords, n_clusters)
    centers = kmeans.cluster_centers_
    m_pred = folium.Map(location=[coords['lat'].mean(), coords['lon'].mean()], zoom_start=4, tiles="CartoDB positron")
    for _, row in coords.sample(n=min(1500, len(coords)), random_state=42).iterrows():
        folium.CircleMarker([row['lat'], row['lon']], radius=1, color="gray", fill=True, fill_opacity=0.4).add_to(m_pred)
    for i, center in enumerate(centers):
        folium.Marker(center, popup=f"Suggested Tower #{i+1}", icon=folium.Icon(color="red", icon="plus")).add_to(m_pred)
    st_folium(m_pred, width=750)
else:
    st.warning("Not enough data points for clustering.")

# --- Reduce spacing between maps ---
st.markdown("<div style='margin-top:-40px'></div>", unsafe_allow_html=True)

# --- 2. AGGREGATOR-WISE RECOMMENDATIONS ---
st.header("🏢 Aggregator-wise New Tower Recommendations")
operators = sorted(df["operator_name"].unique())
operator_sel = st.selectbox("Select Aggregator for Recommendation", operators, key="agg_pred")
df_op = df[df["operator_name"] == operator_sel]
if len(df_op) < 10:
    st.warning("Not enough towers for meaningful clustering for this operator.")
else:
    n_clusters_op = st.slider(f"Clusters for {operator_sel}", min_value=2, max_value=min(15, len(df_op)//5), value=5, key="n_clusters_op")
    coords_op = df_op[['lat', 'lon']].dropna()
    kmeans_op = kmeans_fit(coords_op, n_clusters_op)
    centers_op = kmeans_op.cluster_centers_
    m_op_pred = folium.Map(location=[coords_op['lat'].mean(), coords_op['lon'].mean()], zoom_start=4, tiles="CartoDB positron")
    for _, row in coords_op.sample(n=min(1000, len(coords_op)), random_state=42).iterrows():
        folium.CircleMarker([row['lat'], row['lon']], radius=1, color="blue", fill=True, fill_opacity=0.4).add_to(m_op_pred)
    for i, center in enumerate(centers_op):
        folium.Marker(center, popup=f"Suggested {operator_sel} Tower #{i+1}", icon=folium.Icon(color="red", icon="plus")).add_to(m_op_pred)
    st_folium(m_op_pred, width=700)

st.markdown("<div style='margin-top:-40px'></div>", unsafe_allow_html=True)

# --- 3. TECHNOLOGY-WISE RECOMMENDATIONS ---
st.header("📶 Technology-wise New Tower Recommendations")
techs = sorted(df['radio'].unique())
tech_sel = st.selectbox("Select Technology for Recommendation", techs, key="tech_pred")
df_tech = df[df["radio"] == tech_sel]
if len(df_tech) < 10:
    st.warning("Not enough towers for meaningful clustering for this technology.")
else:
    n_clusters_tech = st.slider(f"Clusters for {tech_sel}", min_value=2, max_value=min(15, len(df_tech)//5), value=5, key="n_clusters_tech")
    coords_tech = df_tech[['lat', 'lon']].dropna()
    kmeans_tech = kmeans_fit(coords_tech, n_clusters_tech)
    centers_tech = kmeans_tech.cluster_centers_
    m_tech_pred = folium.Map(location=[coords_tech['lat'].mean(), coords_tech['lon'].mean()], zoom_start=4, tiles="CartoDB positron")
    for _, row in coords_tech.sample(n=min(1000, len(coords_tech)), random_state=42).iterrows():
        folium.CircleMarker([row['lat'], row['lon']], radius=1, color="orange", fill=True, fill_opacity=0.4).add_to(m_tech_pred)
    for i, center in enumerate(centers_tech):
        folium.Marker(center, popup=f"Suggested {tech_sel} Tower #{i+1}", icon=folium.Icon(color="red", icon="plus")).add_to(m_tech_pred)
    st_folium(m_tech_pred, width=700)

st.markdown("<div style='margin-top:-40px'></div>", unsafe_allow_html=True)
st.markdown("---")

# --- 4. AGGREGATOR & TECHNOLOGY RECOMMENDATION MAP ---
st.header("🗺️ Aggregator + Technology - Specific New Tower Suggestion Map")
st.markdown("Select both an aggregator and a technology to visualize where to put new towers for that specific combo.")

agg_combo = st.selectbox("Aggregator (for Combo Map)", operators, key="combo_agg")
tech_combo = st.selectbox("Technology (for Combo Map)", techs, key="combo_tech")
df_combo = df[(df["operator_name"] == agg_combo) & (df["radio"] == tech_combo)]
if len(df_combo) < 5:
    st.warning("Not enough towers for this aggregator/technology combo for clustering.")
else:
    n_clusters_combo = st.slider(f"Clusters for {agg_combo} ({tech_combo})", min_value=2, max_value=min(8, len(df_combo)), value=min(5, len(df_combo)//3), key="combo_n_clusters")
    coords_combo = df_combo[['lat', 'lon']].dropna()
    kmeans_combo = kmeans_fit(coords_combo, n_clusters_combo)
    centers_combo = kmeans_combo.cluster_centers_
    m_combo = folium.Map(location=[coords_combo['lat'].mean(), coords_combo['lon'].mean()], zoom_start=5, tiles="CartoDB positron")
    # Plot all towers for this operator/tech (blue)
    for _, row in coords_combo.iterrows():
        folium.CircleMarker([row['lat'], row['lon']], radius=2, color="blue", fill=True, fill_opacity=0.6).add_to(m_combo)
    # Plot cluster centers (red)
    for i, center in enumerate(centers_combo):
        folium.Marker(center, popup=f"Suggested {agg_combo}-{tech_combo} Tower #{i+1}", icon=folium.Icon(color="red", icon="plus")).add_to(m_combo)
    st_folium(m_combo, width=750)

st.caption("© 2024 Telecom Tower ML & Visualization App. Built with Streamlit, scikit-learn, Folium.")
