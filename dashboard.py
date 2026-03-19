import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(page_title="Circadian Rhythm Anomaly Detector", layout="wide", page_icon="🧬")

# Custom CSS for UI
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    .stMetric {background-color: #1E2329; padding: 15px; border-radius: 10px;}
    h1, h2, h3 {color: #E0E6ED;}
    </style>
""", unsafe_allow_html=True)

st.title("🧬 Circadian Rhythm Anomaly Detection")
st.markdown("### Real-time Monitoring & Deep Learning Analytics Dashboard")
st.markdown("This dashboard visualizes data analyzed by a lightweight Autoencoder trained to identify disruptions in circadian patterns using wearable sensor data.")

@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "dashboard_data.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None

df = load_data()

if df is not None:
    # Handle NaN values for boolean mapping
    df['is_anomaly'] = df['is_anomaly'].fillna(False).astype(bool)
    
    # ---------------------------------------------------------
    # Analyze Feature Deviations to find "Which Attribute" caused anomalies
    # ---------------------------------------------------------
    anomalies_df = df[df['is_anomaly']]
    normal_df = df[~df['is_anomaly']]
    
    # Normalize with coefficient of variation to compare apples to oranges
    hr_mean_norm = normal_df['HeartRate'].mean()
    hr_std_norm = normal_df['HeartRate'].std()
    hr_std_anom = anomalies_df['HeartRate'].std()
    hr_deviation = (hr_std_anom - hr_std_norm) / hr_std_norm if pd.notnull(hr_std_norm) and hr_std_norm>0 else 0
    
    step_mean_norm = normal_df['StepCount'].mean()
    step_std_norm = normal_df['StepCount'].std()
    step_std_anom = anomalies_df['StepCount'].std()
    step_deviation = (step_std_anom - step_std_norm) / step_std_norm if pd.notnull(step_std_norm) and step_std_norm>0 else 0
    
    primary_anomaly_cause = "Heart Rate" if hr_deviation > step_deviation else "Step Count (Activity)"

    # KPIs
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    total_hours = len(df)
    total_anomalies = len(anomalies_df)
    anomaly_rate = (total_anomalies / total_hours) * 100 if total_hours > 0 else 0
    
    with col1:
        st.metric("Total Monitored Hours", f"{total_hours}")
    with col2:
        st.metric("Detected Anomalies", f"{total_anomalies}")
    with col3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    with col4:
        st.metric("Primary Anomaly Driver", primary_anomaly_cause)

    st.markdown("---")
    
    # ---------------------------------------------------------
    # Digital Doctor's Insights
    # ---------------------------------------------------------
    st.subheader("👨‍⚕️ Digital Doctor's Analysis & Recommendations")
    
    if total_anomalies == 0:
        st.success("**Doctor's Note:** Your circadian rhythm looks exceptionally stable. Keep up your current routine!")
    else:
        st.warning(f"**Doctor's Note:** We detected {total_anomalies} hours of disruption in your circadian cycle. The primary metric responsible for your anomalies is **{primary_anomaly_cause}**.")
        
        # Give tailored suggestions
        st.markdown("**Suggestions to Improve Your Circadian Rhythm:**")
        if primary_anomaly_cause == "Heart Rate":
            st.info("""
            🫀 **Heart Rate Stability:**
            Your heart rate is showing uncharacteristic spikes or drops out of sync with your normal routine. 
            - **Limit Stimulants**: Avoid caffeine, nicotine, or heavy meals at least 4 to 6 hours before bedtime.
            - **Manage Stress**: Incorporate relaxation techniques like meditation or deep breathing exercises into your evening routine.
            - **Hydration & Diet**: Ensure consistent hydration. An irregular heart rate can sometimes be linked to mild dehydration or electrolyte imbalances.
            - **Medical Check**: If resting heart rate anomalies persist, consider discussing them with a physician, as elevated heart rates during rest might indicate cardiovascular stress or poor sleep quality.
            """)
        else:
            st.info("""
            👟 **Activity & Step Count Rhythms:**
            Your physical activity is happening at irregular cyclical times (e.g., late-night activity spikes or unusually prolonged daytime sedentary periods).
            - **Morning Sunlight**: Get 10-15 minutes of natural sunlight first thing in the morning. This signals your brain's biological clock to properly start the day.
            - **Consistent Routine**: Try to wake up and go to sleep at the exact same time every day, even on weekends.
            - **Avoid Evening Workouts**: Keep vigorous exercise, like heavy cardio or lifting, to the morning or late afternoon. Avoid intense activity 2 hours prior to bed.
            - **Break Up Sedentary Time**: If the anomaly is caused by unusual inactivity, remember to stand up or walk for 5 minutes every hour during the day to prevent circadian sluggishness.
            """)

    st.markdown("---")

    # 1. Heart Rate over time with Anomalies
    st.subheader("Heart Rate Variations & Detected Disruptions")
    fig_hr = go.Figure()
    
    # Normal HR line
    fig_hr.add_trace(go.Scatter(
        x=df['timestamp'], y=df['HeartRate'], 
        mode='lines', name='Heart Rate',
        line=dict(color='#00CC96', width=2)
    ))
    
    # Anomalies overlay
    fig_hr.add_trace(go.Scatter(
        x=anomalies_df['timestamp'], y=anomalies_df['HeartRate'],
        mode='markers', name='Anomaly Detected',
        marker=dict(color='#EF553B', size=8, symbol='x')
    ))
    
    fig_hr.update_layout(template="plotly_dark", hovermode="x unified", height=400)
    st.plotly_chart(fig_hr, use_container_width=True)
    
    # 2. Step Count over time
    st.subheader("Activity Levels (Step Count)")
    fig_steps = px.bar(
        df, x='timestamp', y='StepCount',
        color='is_anomaly',
        color_discrete_map={False: '#636EFA', True: '#EF553B'},
        labels={'is_anomaly': 'Anomaly'}
    )
    fig_steps.update_layout(template="plotly_dark", bargap=0, height=400)
    st.plotly_chart(fig_steps, use_container_width=True)
    
    # 3. Reconstruction Error Distribution
    st.markdown("---")
    st.subheader("Deep Learning Reconstruction Error")
    st.markdown("The autoencoder tries to reconstruct the normal 24-hour cycle. When the error exceeds the threshold, an anomaly is flagged.")
    
    fig_error = go.Figure()
    fig_error.add_trace(go.Scatter(
        x=df['timestamp'].dropna(), y=df['reconstruction_error'].dropna(),
        mode='lines', name='Reconstruction Error',
        line=dict(color='#AB63FA')
    ))
    
    # Threshold Line
    threshold_val = df['threshold'].iloc[0]
    fig_error.add_hline(y=threshold_val, line_dash='dash', line_color='red', 
                        annotation_text="Anomaly Threshold", annotation_position="top left")
    
    fig_error.update_layout(template="plotly_dark", hovermode="x unified", height=300)
    st.plotly_chart(fig_error, use_container_width=True)

else:
    st.warning("No data found! Please wait for the Autoencoder script to finish processing `anomaly_detector.py`.")
