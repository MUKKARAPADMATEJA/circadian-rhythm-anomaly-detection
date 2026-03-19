import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Circadian Rhythm Anomaly Detector",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0E1117; }
    .metric-card {
        background: linear-gradient(135deg, #1E2329 0%, #252D3A 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2D3748;
    }
    .doctor-tip {
        background: linear-gradient(135deg, #0d2137 0%, #0a3251 100%);
        border-left: 4px solid #00CC96;
        border-radius: 8px;
        padding: 16px;
    }
    h1 { color: #E0E6ED !important; }
    h2, h3 { color: #A0AEC0 !important; }
    </style>
""", unsafe_allow_html=True)

# ─── MODEL DEFINITION ────────────────────────────────────────────────────────
class LightweightAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=8):
        super(LightweightAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ─── XML PARSER ─────────────────────────────────────────────────────────────
def parse_apple_health_xml(file_obj):
    progress = st.empty()
    records = []
    
    target_hr = "HKQuantityTypeIdentifierHeartRate"
    target_steps = "HKQuantityTypeIdentifierStepCount"
    
    try:
        context = ET.iterparse(file_obj, events=("end",))
        count = 0
        
        for event, elem in context:
            if elem.tag == "Record":
                rec_type = elem.get("type", "")
                if rec_type in [target_hr, target_steps]:
                    date_str = elem.get("startDate") or elem.get("creationDate")
                    value = elem.get("value")
                    if date_str and value:
                        records.append({
                            'timestamp': date_str,
                            'type': 'HeartRate' if rec_type == target_hr else 'StepCount',
                            'value': float(value)
                        })
                elem.clear()
            
            count += 1
            if count % 20000 == 0:
                progress.info(f"⏳ Processing: Read {count:,} XML records... ({len(records):,} target samples found)")
        
        progress.empty()
        if not records:
            st.error("❌ No HeartRate or StepCount data found.")
            return None
            
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        df_pivot = df.pivot_table(index='timestamp', columns='type', values='value', aggfunc='mean')
        df_resampled = df_pivot.resample('1H').mean()
        
        if 'HeartRate' in df_resampled.columns:
            df_resampled['HeartRate'] = df_resampled['HeartRate'].ffill().bfill()
        else:
            df_resampled['HeartRate'] = 70.0
            
        if 'StepCount' in df_resampled.columns:
            df_resampled['StepCount'] = df_resampled['StepCount'].fillna(0)
        else:
            df_resampled['StepCount'] = 0.0
            
        return df_resampled.fillna(0).reset_index()
    except Exception as e:
        st.error(f"❌ XML Error: {e}")
        return None

# ─── INFERENCE ENGINE ─────────────────────────────────────────────────────────
def run_anomaly_inference(df):
    status = st.status("🧠 Running Deep Learning Anomaly Detection...")
    
    input_df = df[['HeartRate', 'StepCount']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(input_df)
    
    window_size = 24
    sequences = []
    for i in range(len(scaled_data) - window_size + 1):
        sequences.append(scaled_data[i: i + window_size])
    sequences = np.array(sequences)
    
    if len(sequences) == 0:
        status.update(label="❌ Total data recorded is less than 24 hours.", state="error")
        return None
        
    input_dim = window_size * 2
    X = torch.tensor(sequences.reshape(-1, input_dim), dtype=torch.float32)
    
    model = LightweightAutoencoder(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for _ in range(30):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), X)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds = model(X)
        errors = torch.mean((preds - X) ** 2, dim=1).numpy()
    
    threshold = float(np.percentile(errors, 95))
    results_df = df.iloc[:len(errors)].copy()
    results_df['reconstruction_error'] = errors
    results_df['is_anomaly'] = errors > threshold
    results_df['threshold'] = threshold
    
    status.update(label="✅ Analysis Complete!", state="complete")
    return results_df

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.title("🧬 Circadian Rhythm Anomaly Detector")
st.markdown("**Real-time Monitoring & Deep Learning Analytics Dashboard**")
st.markdown("A lightweight autoencoder trained on wearable sensor data to detect circadian rhythm disruptions.")
st.markdown("---")

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2491/2491528.png", width=80)
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📂 Upload `export.xml` or `dashboard_data`",
        help="Upload raw Apple Health XML or generated dashboard_data.csv (up to 2GB supported)."
    )
    
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown("""
    - **Model**: Lightweight Autoencoder (PyTorch)  
    - **Threshold**: 95th Percentile Reconstruction Error  
    - **Data**: Apple Watch Heart Rate & Step Count  
    - **Window**: 24-hour circadian cycle  
    """)
    st.markdown("---")
    st.markdown("**SDGs:** 🌱 Goal 3 · ⚙️ Goal 9")

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data_from_file(file):
    try:
        if file is None: return None
        file_name = file.name.lower()
        file.seek(0)
        
        if file_name.endswith('.xml'):
            raw_df = parse_apple_health_xml(file)
            if raw_df is not None:
                return run_anomaly_inference(raw_df)
            return None
            
        header_sample = file.read(1024 * 10).decode('utf-8', errors='ignore').splitlines()
        file.seek(0)
        
        header_row = 0
        found_header = False
        for i, line in enumerate(header_sample[:50]):
            if 'timestamp' in line.lower() and ('reduction' in line.lower() or 'anomaly' in line.lower() or 'heartrate' in line.lower()):
                header_row = i
                found_header = True
                break
        
        df = pd.read_csv(file, skiprows=header_row) if found_header else pd.read_csv(file, on_bad_lines='skip')
        if 'timestamp' not in df.columns:
            st.error("❌ Invalid CSV: Missing 'timestamp' column.")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df.dropna(subset=['timestamp']).sort_values('timestamp')
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

@st.cache_data
def load_local_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "dashboard_data.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df.sort_values('timestamp').dropna(subset=['timestamp'])
    return None

# Load logic
if uploaded_file is not None:
    df = load_data_from_file(uploaded_file)
else:
    df = load_local_data()

# ─── MAIN DASHBOARD ───────────────────────────────────────────────────────────
if df is not None:
    if 'is_anomaly' in df.columns:
        df['is_anomaly'] = df['is_anomaly'].fillna(False).astype(bool)
        anomalies_df = df[df['is_anomaly']]
        normal_df = df[~df['is_anomaly']]
        threshold_val = df['threshold'].iloc[0] if 'threshold' in df.columns else 0.05

        # Metrics
        total_hours = len(df)
        total_anomalies = len(anomalies_df)
        anomaly_rate = (total_anomalies / total_hours) * 100 if total_hours > 0 else 0
        
        # Attribution
        hr_std_norm = normal_df['HeartRate'].std() if 'HeartRate' in normal_df.columns else 1
        hr_std_anom = anomalies_df['HeartRate'].std() if len(anomalies_df) > 0 else 0
        hr_deviation = (hr_std_anom - hr_std_norm) / hr_std_norm if hr_std_norm > 0 else 0
        
        step_std_norm = normal_df['StepCount'].std() if 'StepCount' in normal_df.columns else 1
        step_std_anom = anomalies_df['StepCount'].std() if len(anomalies_df) > 0 else 0
        step_deviation = (step_std_anom - step_std_norm) / step_std_norm if step_std_norm > 0 else 0
        
        primary_cause = "Heart Rate" if hr_deviation > step_deviation else "Step Count"

        # KPI Metrics Row
        st.subheader("📊 Key Performance Indicators")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🕐 Monitored Hours", f"{total_hours:,}")
        c2.metric("⚠️ Anomalies Detected", f"{total_anomalies:,}")
        c3.metric("📈 Anomaly Rate", f"{anomaly_rate:.2f}%")
        c4.metric("🔑 Primary Driver", primary_cause)
        c5.metric("🤖 Detection Mode", "AI Inference")

        st.markdown("---")
        
        # Doctor recommendations
        st.subheader("👨‍⚕️ Digital Doctor's Analysis & Recommendations")
        st.warning(f"⚠️ **Doctor's Note:** We detected **{total_anomalies} hours** of circadian disruption. Primary driver: **{primary_cause}**.")
        
        cd1, cd2 = st.columns(2)
        with cd1:
            st.markdown("##### 🫀 Heart Rate Recommendations")
            st.info("Limit caffeine 6h before bed | Practice box breathing | Monitor resting HR.")
        with cd2:
            st.markdown("##### 👟 Activity Recommendations")
            st.info("Consistent wake time | Morning sunlight exposure | Avoid vigorous night exercise.")

        # Timeline
        st.subheader("❤️ Heart Rate Timeline")
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(x=df['timestamp'], y=df['HeartRate'], mode='lines', name='Heart Rate', line=dict(color='#00CC96')))
        fig_hr.add_trace(go.Scatter(x=anomalies_df['timestamp'], y=anomalies_df['HeartRate'], mode='markers', name='Anomaly', marker=dict(color='#EF553B', size=8, symbol='x')))
        fig_hr.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_hr, use_container_width=True)
    else:
        st.warning("Data loaded, but missing anomaly analysis. Please ensure you upload the correct file.")
else:
    st.info("👈 **Upload your raw `export.xml` or `dashboard_data.csv`** in the sidebar to begin analysis.")
