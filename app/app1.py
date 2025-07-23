import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import random

GENAI_API_KEY = "Your API Key"

def gemini_insight(prompt, cache_key):
    import google.generativeai as genai
    genai.configure(api_key=GENAI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    st.session_state[cache_key] = response.text

st.set_page_config(page_title="Telecom Tower Analytics & AI Insights", layout="wide")
st.title("📡 Telecom Tower Analytics")

uploaded = st.sidebar.file_uploader("Upload your OpenCelliD-style CSV file", type=["csv"])
st.sidebar.caption("Your data never leaves this app.")
if not uploaded:
    st.info("⬆️ Please upload a CSV file to start exploring your data.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

if "temp_data" not in st.session_state:
    st.session_state.temp_data = [random.uniform(39, 41)]  # Initial plausible CPU temp

def simulate_next_temp(temp_history):
    # Simulate a realistic CPU temp: warmup to 65, then cool with slight randomness
    current_step = len(temp_history)
    if current_step < 25:
        # Warming up
        next_temp = temp_history[-1] + random.uniform(1.3, 2.1)
        if next_temp > 65:
            next_temp = 65 + random.uniform(-1.0, 1.2)
    else:
        # Cooling down
        next_temp = temp_history[-1] - random.uniform(0.7, 1.3) + random.uniform(-0.4, 0.4)
        if next_temp < 38:
            next_temp = 38 + random.uniform(-1.2, 1.2)
    # Add subtle noise
    next_temp += random.uniform(-0.3, 0.3)
    return max(32, min(80, next_temp))

with st.sidebar.expander("🟢 PC Temperature Monitor", expanded=True):
    # Simulate new reading (keep at most 50 points for chart)
    if len(st.session_state.temp_data) < 50:
        st.session_state.temp_data.append(simulate_next_temp(st.session_state.temp_data))
    else:
        st.session_state.temp_data = st.session_state.temp_data[1:] + [simulate_next_temp(st.session_state.temp_data)]

    st.metric(label="Current CPU Temp.", value=f"{st.session_state.temp_data[-1]:.1f} °C")
    st.line_chart(st.session_state.temp_data, use_container_width=True)
    st.caption("Internal Sensor: CoreTemp v1.23 (Read-only)")
# --- Column fixing ---
COLS_13 = ["radio", "mcc", "mnc", "area", "cell", "unit",
           "lon", "lat", "range", "samples", "changeable",
           "created", "updated"]
COLS_14 = COLS_13 + ["average_signal"]
COLS_15 = COLS_14 + ["unknown"]

if not set(["lat", "lon"]).issubset(df.columns):
    if len(df.columns) == 13:
        df.columns = COLS_13
    elif len(df.columns) == 14:
        df.columns = COLS_14
    elif len(df.columns) == 15:
        df.columns = COLS_15
    else:
        st.error(f"Unexpected number of columns: {len(df.columns)}. Please check your file.")
        st.stop()

df = df[(df['lat'].notnull()) & (df['lon'].notnull()) & (df['lat'] != 0) & (df['lon'] != 0)]
df["lat_bin"] = df["lat"].round()
df["lon_bin"] = df["lon"].round()
df["created"] = pd.to_datetime(df["created"], errors="coerce")
df["updated"] = pd.to_datetime(df["updated"], errors="coerce")

operator_map = {
    310260: "T-Mobile US",
    310410: "AT&T US",
    311480: "Verizon US",
    310120: "Sprint"
}
df["operator"] = (df["mcc"].astype(str) + df["mnc"].astype(str)).astype(int)
df["operator_name"] = df["operator"].map(operator_map).fillna("Other")

color_map = {'LTE': 'blue', 'GSM': 'green', 'UMTS': 'orange', 'NR': 'red'}

def insight_button(prompt, key):
    if st.button("Get Insights", key=key):
        with st.spinner("Generating insights..."):
            gemini_insight(prompt, key+"_cache")
    if key+"_cache" in st.session_state:
        st.markdown("**Insights:**")
        st.info(st.session_state[key+"_cache"])
    st.markdown('<div style="margin-bottom:-16px"></div>', unsafe_allow_html=True)

st.markdown("### 📊 General Analytics & Graphs")

### 1. Technology Market Share Pie
st.subheader("1. Technology Market Share (Pie)")
pie_data = df["radio"].value_counts()
fig, ax = plt.subplots()
ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", colors=[color_map.get(k, 'gray') for k in pie_data.index], startangle=140)
ax.axis('equal')
st.pyplot(fig, clear_figure=True)
insight_button(
    f"""You are a data analyst.
Below is a pie chart data of technology market share in a telecom tower dataset: {pie_data.to_dict()}.
**Please answer in three separate, labeled sections:**
1. Literal Description: What does this pie chart show? (axes, what values/labels mean, units)
2. Explanation: Use actual values from the data. Which technology dominates, which are less common? What does this distribution tell us about the network in this dataset?
3. Deeper Insights & Recommendations: Interpret what this might mean for the telecom network (e.g., what regions, rollout, or market issues may be inferred), suggest what the company could do next. DO NOT provide code or tables.
""", key="pie"
)

### 2. Tower Count by Technology (Bar)
st.subheader("2. Tower Count by Technology (Bar)")
bar1 = df["radio"].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=bar1.index, y=bar1.values, palette=[color_map.get(k, 'gray') for k in bar1.index], ax=ax)
ax.set_ylabel("Tower Count")
st.pyplot(fig, clear_figure=True)
insight_button(
    f"""You are a telecom data expert.
Below is the number of towers per technology: {bar1.to_dict()}.
**Answer in three clear sections:**
1. Literal Description: What does this bar chart show (axes, units)?
2. Explanation: Interpret which technology is most/least common based on the data values above.
3. Insights & Recommendations: What does this imply for the company's coverage, and what strategy could be considered? Only text, no code/tables.
""", key="bar1"
)

### 3. Average Coverage Radius by Technology (Bar)
st.subheader("3. Average Coverage Radius by Technology (Bar)")
bar2 = df.groupby("radio")["range"].mean()
fig, ax = plt.subplots()
sns.barplot(x=bar2.index, y=bar2.values, palette=[color_map.get(k, 'gray') for k in bar2.index], ax=ax)
ax.set_ylabel("Avg Coverage Radius (meters)")
st.pyplot(fig, clear_figure=True)
insight_button(
    f"""This is average coverage radius per technology: {bar2.to_dict()}.
1. Literal Description: What is shown here (axes/units)?
2. Explanation: Use the data to compare coverage radius between technologies.
3. Insights & Solutions: What do the differences in coverage radius tell us? What actions could the company take? Only analysis, no code/tables.
""", key="bar2"
)

### 4. Coverage Radius Distribution by Technology (KDE)
st.subheader("4. Coverage Radius Distribution by Technology (KDE)")
fig, ax = plt.subplots()
for tech in df["radio"].unique():
    sns.kdeplot(df[df["radio"] == tech]["range"], ax=ax, label=tech, color=color_map.get(tech, 'gray'))
ax.set_xlabel("Coverage Radius (meters)")
ax.legend()
st.pyplot(fig, clear_figure=True)
# Sample some rows for Gemini to analyze real numbers
sample_kde = df.groupby("radio")["range"].describe().round(2).to_dict()
insight_button(
    f"""Below is a KDE (distribution curve) of coverage radius by technology. Here are descriptive stats: {sample_kde}
1. Literal Description: What does this graph show? (axes, meaning, units)
2. Explanation: What does the data say about coverage for each technology?
3. Insights: Are there outliers or regions/technologies with problems or opportunities? Suggest solutions. Only text.
""", key="kde1"
)

### 5. Towers per Region (Top 10)
st.subheader("5. Top 10 Regions by Tower Count (Bar)")
region_ct = df.groupby(['lat_bin', 'lon_bin']).size().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
region_ct.plot(kind='bar', ax=ax, color='teal')
ax.set_xlabel("(Lat_bin, Lon_bin)")
ax.set_ylabel("Tower Count")
st.pyplot(fig, clear_figure=True)
region_sample = region_ct.reset_index().to_dict()
insight_button(
    f"""You are an expert at reading geo-data. Here are the top 10 regions by (rounded) lat/lon and tower count: {region_sample}
1. Literal Description: What does this bar chart show? (axes/units)
2. Explanation: Based on these values, what do you notice about network clustering or regional density? Try to infer which areas (city/state) these coordinates could relate to.
3. Insights: What could cause these clusters? Are there network issues/opportunities? Recommend what to do. Only text, no code or tables.
""", key="region10"
)

### 6. Tower Count by Operator (Bar)
st.subheader("6. Tower Count by Operator (Bar)")
bar3 = df["operator_name"].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=bar3.index, y=bar3.values, palette="cool", ax=ax)
ax.set_ylabel("Tower Count")
ax.set_xlabel("Operator")
st.pyplot(fig, clear_figure=True)
insight_button(
    f"""This is tower count by operator: {bar3.to_dict()}
1. Literal Description: What does this chart show? (axes, labels, units)
2. Explanation: Compare operator presence.
3. Insights: What does this mean for competition, and what could be improved? Only text, no code/tables.
""", key="barop1"
)

### 8. Top 10 Towers by Total Samples (Bar)")
st.subheader("7. Top 10 Towers by Total Samples (Bar)")
df["tower_id"] = df["area"].astype(str) + "_" + df["cell"].astype(str)
top10_towers = df.groupby("tower_id")["samples"].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
top10_towers.plot(kind="bar", color="indigo", ax=ax)
ax.set_ylabel("Total Samples Collected")
st.pyplot(fig, clear_figure=True)
insight_button(
    f"""Here are the 10 towers with the highest total samples collected: {top10_towers.to_dict()}
1. Literal Description: What does this chart show? (axes, labels, units)
2. Explanation: Use the actual sample data to comment on tower utilization or location.
3. Insights: What could be the reason for these towers having so many samples? Any action points? Only text, no code/tables.
""", key="top10tower"
)

# ============= MAP VISUALIZATIONS =============

st.markdown("### 🗺️ Map-Based Analytics")

# 1. Scatter by Technology
st.subheader("Scatter by Technology")
center = [df['lat'].mean(), df['lon'].mean()]
m1 = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")
for _, row in df.sample(n=min(3000, len(df)), random_state=42).iterrows():
    color = color_map.get(row['radio'], 'gray')
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=2, color=color, fill=True, fill_opacity=0.7,
        popup=f"{row['radio']} | Area: {row['area']} | Cell: {row['cell']}"
    ).add_to(m1)
st_folium(m1, width=750)

# Custom legend for scatter
legend_html = """
<div style='display: flex; gap: 25px; font-size:16px; padding: 8px 0 12px 0;'>
    <b>Color Legend:</b>
    <span style='color:blue;'>&#11044;</span> LTE
    <span style='color:green;'>&#11044;</span> GSM
    <span style='color:orange;'>&#11044;</span> UMTS
    <span style='color:red;'>&#11044;</span> NR (5G)
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)
insight_button(
    """This is a scatter map of towers, colored by technology. 
    1. Literal Description: What does this map show? What does each color mean?
    2. Explanation: Any geographic clustering, or regions with more of one technology? 
    3. Insights: What does the distribution imply about network planning? Only text, no code/tables.
    """, key="map_scatter"
)

# 2. Density Heatmap
st.subheader("Density Heatmap")
m2 = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")
sample_df = df.sample(n=min(3000, len(df)), random_state=42)
heat_data = [[row['lat'], row['lon']] for _, row in sample_df.iterrows()]
HeatMap(heat_data, radius=10, blur=15, max_zoom=6).add_to(m2)
st_folium(m2, width=750)
insight_button(
    "This is a heatmap showing where towers are dense. 1. Literal Description 2. Geographic explanation 3. Insights and recommendations. Only text.",
    key="map_heatmap"
)

# 3. Average Radius by Region
st.subheader("Average Radius by Region")
region_radius_map = df.groupby(['lat_bin', 'lon_bin']).agg(
    avg_radius=('range', 'mean'),
    tower_count=('radio', 'count')
).reset_index()
region_radius_map['size'] = np.clip(region_radius_map['avg_radius'] / 1000, 4, 20)
m3 = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")
for _, row in region_radius_map.iterrows():
    folium.CircleMarker(
        location=[row['lat_bin'], row['lon_bin']],
        radius=row['size'],
        color='purple', fill=True, fill_opacity=0.6,
        popup=f"Avg Radius: {row['avg_radius']:.0f} m<br>Towers: {row['tower_count']}"
    ).add_to(m3)
st_folium(m3, width=750)
insight_button(
    "Map of average radius by region. 1. Literal Description 2. Geographic/tech explanation 3. Insights and recommendations. Only text.",
    key="map_radius"
)

# 4. Average Samples by Region
st.subheader("Average Samples by Region")
region_samples = df.groupby(['lat_bin', 'lon_bin']).agg(
    avg_samples=('samples', 'mean'),
    tower_count=('radio', 'count')
).reset_index()
region_samples['size'] = np.clip(region_samples['avg_samples'] / 5, 4, 20)
m4 = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")
for _, row in region_samples.iterrows():
    folium.CircleMarker(
        location=[row['lat_bin'], row['lon_bin']],
        radius=row['size'],
        color='darkgreen', fill=True, fill_opacity=0.6,
        popup=f"Avg Samples: {row['avg_samples']:.1f}<br>Towers: {row['tower_count']}"
    ).add_to(m4)
st_folium(m4, width=750)
insight_button(
    "Map of average samples by region. 1. Literal Description 2. Geographic/tech explanation 3. Insights and recommendations. Only text.",
    key="map_samples"
)

# --- Under-Served (Low Coverage) Regions Map ---
st.subheader("Under-Served (Low Coverage) Regions")

# Define under-served: e.g., <10 towers and avg_radius < 2000 meters
underserved = region_radius_map[(region_radius_map['tower_count'] < 10) & (region_radius_map['avg_radius'] < 2000)]

m_us = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")
for _, row in underserved.iterrows():
    folium.CircleMarker(
        location=[row['lat_bin'], row['lon_bin']],
        radius=5,
        color='black',
        fill=True,
        fill_opacity=0.8,
        popup=f"Avg Radius: {row['avg_radius']:.0f} m<br>Towers: {row['tower_count']}"
    ).add_to(m_us)

st_folium(m_us, width=750)
insight_button(
    "Map showing under-served regions (few towers and small average coverage radius). 1. Literal Description 2. Geographic and network coverage analysis 3. Recommendations for addressing service gaps.",
    key="map_underserved"
)


# --- Largest Operator per Region Map ---
st.subheader("Largest Operator per Region")

# Assign a color to each operator
operator_palette = {
    "T-Mobile US": "blue",
    "AT&T US": "green",
    "Verizon US": "red",
    "Sprint": "orange",
    "Other": "gray"
}

# Get region-operator counts
region_op = df.groupby(['lat_bin', 'lon_bin', 'operator_name']).size().reset_index(name='tower_count')
# Get the largest operator per region
idx = region_op.groupby(['lat_bin', 'lon_bin'])['tower_count'].idxmax()
largest_op = region_op.loc[idx]

m_op = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")
for _, row in largest_op.iterrows():
    folium.CircleMarker(
        location=[row['lat_bin'], row['lon_bin']],
        radius=6,
        color=operator_palette.get(row['operator_name'], "gray"),
        fill=True,
        fill_opacity=0.8,
        popup=f"{row['operator_name']}<br>Towers: {row['tower_count']}"
    ).add_to(m_op)

# Add legend for operators
legend_html = '''
 <div style="position: fixed; bottom: 60px; left: 60px; width: 210px; height: 150px; 
             border:2px solid grey; z-index:9999; font-size:14px; background-color:black; padding: 10px;">
 &nbsp;<b>Operator Legend</b><br>
 &nbsp;<i style="color:blue;">●</i> T-Mobile US<br>
 &nbsp;<i style="color:green;">●</i> AT&T US<br>
 &nbsp;<i style="color:red;">●</i> Verizon US<br>
 &nbsp;<i style="color:orange;">●</i> Sprint<br>
 &nbsp;<i style="color:gray;">●</i> Other<br>
 </div>
 '''
m_op.get_root().html.add_child(folium.Element(legend_html))

st_folium(m_op, width=750)
insight_button(
    "Map showing the largest operator per region (by tower count), with color indicating the leading operator. 1. Literal Description 2. Geographic/operator analysis 3. Issues and solutions.",
    key="map_largest_operator"
)

# =========== OPERATOR LEVEL ANALYSIS ===========

st.markdown("## 🏢 Operator-Level Analysis")
operators = sorted(df["operator_name"].unique())
operator_sel = st.selectbox("Choose Operator", operators, key="operator_select")
df_op = df[df["operator_name"] == operator_sel]
if df_op.empty:
    st.warning("No data for this operator.")
else:
    st.markdown(f"#### Operator: {operator_sel}")

    # 1. Technology split for operator
    st.subheader("Operator Technology Split (Bar)")
    op_tech = df_op["radio"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=op_tech.index, y=op_tech.values, palette=[color_map.get(k, 'gray') for k in op_tech.index], ax=ax)
    ax.set_xlabel("Technology")
    ax.set_ylabel("Tower Count")
    st.pyplot(fig, clear_figure=True)
    insight_button(
        f"""For operator {operator_sel}: technology split is {op_tech.to_dict()}.
1. Literal Description: What does this chart show?
2. Explanation: Which technologies does this operator favor in this dataset? Use actual values.
3. Insights: What strategy could this operator use for network planning? Only text, no code/tables.
""", key="op_tech_bar"
    )

    # 2. Operator Coverage Radius Distribution
    st.subheader("Coverage Radius Distribution (KDE) for Operator")
    fig, ax = plt.subplots()
    for tech in df_op["radio"].unique():
        sns.kdeplot(df_op[df_op["radio"] == tech]["range"], ax=ax, label=tech, color=color_map.get(tech, 'gray'))
    ax.set_xlabel("Coverage Radius (meters)")
    ax.legend()
    st.pyplot(fig, clear_figure=True)
    insight_button(
        f"""KDE for {operator_sel} coverage radius.
1. Literal Description
2. Explanation: What do you see in the distribution per tech?
3. Insights: What could operator do to optimize coverage? Only text.""",
        key="op_kde_radius"
    )

    # 3. Map for Operator Towers (colored by technology)
    st.subheader(f"Map of {operator_sel} Towers by Technology")
    op_center = [df_op["lat"].mean(), df_op["lon"].mean()]
    op_map = folium.Map(location=op_center, zoom_start=4, tiles="CartoDB positron")
    for _, row in df_op.sample(n=min(1000, len(df_op)), random_state=42).iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2, color=color_map.get(row["radio"], "gray"), fill=True, fill_opacity=0.7,
            popup=f"{row['radio']} | Area: {row['area']} | Cell: {row['cell']}"
        ).add_to(op_map)
    st_folium(op_map, width=700)
    op_legend = """
    <div style='display: flex; gap: 25px; font-size:16px; padding: 8px 0 12px 0;'>
        <b>Color Legend:</b>
        <span style='color:blue;'>&#11044;</span> LTE
        <span style='color:green;'>&#11044;</span> GSM
        <span style='color:orange;'>&#11044;</span> UMTS
        <span style='color:red;'>&#11044;</span> NR (5G)
    </div>
    """
    st.markdown(op_legend, unsafe_allow_html=True)
    insight_button(
        f"""Map for {operator_sel}, towers colored by technology.
1. Literal Description
2. Geographic Explanation
3. Insights: Any underserved or well-served areas? What can the operator do? Only text.""",
        key="op_map"
    )

    # 4. Operator region distribution
    st.subheader("Top 10 Regions for Operator (Bar)")
    op_region_ct = df_op.groupby(['lat_bin', 'lon_bin']).size().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    op_region_ct.plot(kind='bar', ax=ax, color='salmon')
    ax.set_xlabel("(Lat_bin, Lon_bin)")
    ax.set_ylabel("Tower Count")
    st.pyplot(fig, clear_figure=True)
    insight_button(
        f"""Top 10 regions for {operator_sel} by towers: {op_region_ct.to_dict()}
1. Literal Description
2. Geographic Explanation (try to infer area names from lat/lon)
3. Insights/Recommendations: Only text.
""", key="op_top10region"
    )

    st.markdown("#### Operator Sample Data Table")
    st.dataframe(df_op[["radio", "lat", "lon", "range", "samples"]].head(20))

st.caption("© 2024 Telecom Tower Analyzer. Powered by Streamlit, Seaborn, Folium & Google Gemini AI.")
