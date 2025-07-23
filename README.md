# 📡 Geospatial Telecom Infrastructure Optimization

## 🚀 Project Overview

Modern society relies on robust telecom infrastructure to stay connected. However, traditional tower planning is slow and leads to both over-served (wasteful) and under-served (excluded) regions—especially in rural or rapidly growing areas.

This project demonstrates how big data, machine learning, and geospatial analysis can modernize telecom tower planning for better digital inclusion and cost-efficiency.

Using the OpenCelliD dataset (millions of tower records), our solution:
- Cleans and preprocesses raw data
- Maps coverage and technology share
- Analyzes density and identifies coverage gaps
- Uses ML (K-Means) to suggest new tower sites
- Compares operator performance
- Provides interactive dashboards and detailed analytics for both business and technical audiences

The solution integrates Streamlit dashboards, PySpark/MapReduce scripts, and folium-based interactive maps.


## 📂 Project Structure & File-by-File Explanation

```
.
├── app/
│   ├── app1.py          # Interactive analytics dashboard
│   ├── app2.py             # ML-driven recommendations & advanced analytics
│   ├── tower_all_layers_map.html   # Exported folium interactive map (multi-layer)
├── bigdata/
│   ├── mapreduce.py                # Hadoop MapReduce-style analytics
│   ├── sparkan.py                  # Advanced analytics using PySpark DataFrames
├── data/
│   └── 310.csv                     # Sample OpenCelliD-format data
├── requirements.txt                # Python dependencies for the full stack
└── README.md                       # You are here!
```

### Detailed File Explanations

#### `app/streamlit_basic.py`
- Main Streamlit dashboard for tower analytics
- Lets users upload OpenCelliD-style CSV files
- Interactive graphs: technology share pie charts, operator breakdowns, location heatmaps
- Live CPU temperature simulation (demonstrates real-time features)
- "Get Insights" button leverages Google Gemini API for automated chart commentary
- Designed for exploratory analysis, not just ML

#### `app/streamlit_ml.py`
- Advanced Streamlit dashboard focused on predictive analytics
- Uses K-Means clustering to suggest optimal new tower sites based on coverage gaps
- Operator-wise and technology-wise clustering—finds best regions for expansion per operator/technology
- Interactive folium maps visualize current and recommended tower locations
- Allows scenario analysis by adjusting cluster count (number of new towers)

#### `app/tower_all_layers_map.html`
- Exported multi-layer interactive map (HTML, openable in any browser)
- Shows tower technologies, operator overlays, region statistics
- Great for sharing static visualizations outside Streamlit

#### `bigdata/mapreduce.py`
- Hadoop-style MapReduce script (PySpark RDDs) for analysis at scale
- Steps:
  - Loads and preprocesses dataset
  - Counts total towers and technologies
  - Computes towers per region (rounded lat/lon)
  - Aggregates by radio type, computes average range
  - Finds regions/towers with the most/least coverage
  - Identifies unique cell IDs, high-sample towers, and more
- CLI output: designed for learning, research, and rapid batch analytics

#### `bigdata/sparkan.py`
- Advanced analytics with PySpark DataFrames
- Performs:
  - Technology distribution and coverage
  - Region summary (tower count, avg range, avg samples)
  - Temporal analysis (towers by year, by technology)
  - Operator share breakdown
  - Finds under-served regions, top towers, and more
- Outputs rich summaries for dashboards, academic use, or further analysis

#### `bigdata/README.md`
- Instructions for running MapReduce and PySpark scripts (cluster or local mode)
- Troubleshooting tips for PySpark

#### `data/310.csv`
- Sample dataset (anonymized OpenCelliD format)
- Includes columns: radio, mcc, mnc, area, cell, lon, lat, range, samples, created, updated, etc.
- (Do not upload private or proprietary data to GitHub! For demonstration/sample use only.)

#### `requirements.txt`
- All dependencies for Streamlit apps and PySpark scripts
- See [Installation & Usage](#installation--usage) for details

## ✨ Features

- Interactive data upload and analytics via Streamlit
- Coverage heatmaps, technology/operator share charts, and regional summaries
- Big data processing with PySpark/MapReduce for scalability
- Automated insights using generative AI (Gemini)
- ML-driven recommendations for new tower locations
- Operator-wise and technology-wise clustering
- All visualizations are reproducible and customizable
- Exportable maps for sharing results with non-technical stakeholders

## 🛠 Technologies Used

- **Python 3.x**
- **Streamlit** (dashboard/web app)
- **pandas, numpy, matplotlib, seaborn** (EDA, charts)
- **folium** (interactive mapping)
- **PySpark** (big data processing, DataFrames & RDDs)
- **scikit-learn** (K-Means clustering)
- **Google Gemini GenerativeAI API** (automated chart explanations)
- **OpenCelliD data** (or similar format)

## ⚡ Installation & Usage

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/telecom-tower-analytics.git
cd telecom-tower-analytics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API secrets
For Google Gemini or other keys:

**Use Streamlit's secrets.toml:**
`app/.streamlit/secrets.toml`:
```ini
GENAI_API_KEY = "your_google_genai_key"
```

**In code, access as:**
```python
import streamlit as st
key = st.secrets["GENAI_API_KEY"]
```

Or use `.env` and `os.environ`, but do not commit your secrets!

### 4. Run the dashboards
```bash
streamlit run app/streamlit_basic.py
# or
streamlit run app/streamlit_ml.py
```

### 5. Run Big Data scripts
```bash
cd bigdata
python sparkan.py      # PySpark advanced analytics
python mapreduce.py    # MapReduce demo
```

## 🔑 How Secrets and API Keys are Managed

- No API keys are ever hardcoded.
- Use `.env` (for local dev) or `.streamlit/secrets.toml` for cloud/Streamlit sharing.
- Both files are in `.gitignore` so secrets never leak.
- All code is written to securely load keys.

## 📈 Data

The main dataset (`310.csv`) uses OpenCelliD fields:
- `radio`, `mcc`, `mnc`, `area`, `cell`, `lon`, `lat`, `range`, `samples`, `created`, `updated`, etc.
- You may substitute your own data (with matching format).
- **Important:** Do not push or upload real user or proprietary data to public repositories.

## 📚 References & Further Reading

**Literature Review** (see `docs/project_report.docx` for full details):
- Musa, I., et al. (2024): GIS viewshed analysis of cellular towers
- Al‑Hamami, A. H., & Hashem, S. H. (2011): Spatial mining for tower placement (arXiv:1104.2721)
- Bharadwaj, S., et al. (2023): LiDAR data for urban tower planning
- Zhang, X., et al. (2024): Hybrid simulated annealing for tower siting

**Dataset:**
- [OpenCelliD – The world's largest open database of cell towers](https://opencellid.org/)
