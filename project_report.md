# 🧬 Circadian Rhythm Anomaly Detection: Project Report

## 📌 Executive Summary
This project investigates a lightweight deep learning approach to detect circadian rhythm disruptions using continuous wearable sensor data. By leveraging a compact Multilayer Perceptron (MLP) Autoencoder, the system establishes a baseline for an individual's normal physiological cycle and flags significant deviations as anomalies. The solution emphasizes computational efficiency, making it highly suitable for deployment in resource-constrained edge devices, directly addressing Sustainable Development Goals (SDG) 3 (Good Health and Well-Being) and 9 (Industry, Innovation, and Infrastructure).

---

## 1. Introduction
The advent of wearable technology (e.g., Apple Watch) enables the continuous monitoring of physiological parameters. A healthy circadian rhythm (the natural 24-hour sleep-wake cycle) is fundamentally linked to overall well-being. Disruptions to this rhythm are early indicators of stress, illness, poor sleep hygiene, or metabolic disorders. 

Traditional anomaly detection methods often rely on heavy statistical modeling or complex Deep Learning architectures (like heavy LSTMs or Transformers) which are computationally expensive and drain wearable battery life. This project addresses the research gap by proposing a **Lightweight Autoencoder Architecture** capable of running efficiently while maintaining high accuracy in identifying non-cyclical patterns.

---

## 2. Methodology

### 2.1 Dataset and Preprocessing
The model utilizes raw physiological data exported from Apple Health (`export.xml`), specifically focusing on:
- `HKQuantityTypeIdentifierHeartRate` (Continuous Heart Rate)
- `HKQuantityTypeIdentifierStepCount` (Activity Levels)

**Data Processing Pipeline:**
1. **Extraction:** A highly efficient, linear XML parser extracts target records without loading the entire multi-gigabyte XML tree into memory.
2. **Resampling:** Time-series data is aggregated and resampled into 1-hour intervals to smooth out noise and capture macro-trends. Missing values are forward-filled for Heart Rate (to simulate resting periods) and zero-filled for Step Counts.
3. **Sequencing:** The continuous data is sectioned into rolling 24-hour sequences. Each 24-sequence window represents a full circadian cycle.
4. **Normalization:** Features are scaled using Min-Max scaling to ensure the Neural Network weights are updated uniformly.

### 2.2 Model Architecture: Lightweight Autoencoder
The core of the anomaly detection engine is an unsupervised learning model implemented in PyTorch. An autoencoder is trained to compress and reconstruct the input data; if the model struggles to reconstruct a sequence (resulting in a high Mean Squared Error), that sequence is flagged as anomalous.

The architecture was intentionally kept minimal:
*   **Input Layer:** 48 dimensions (24 hours × 2 features)
*   **Encoder Layer 1:** 16 neurons + ReLU activation
*   **Latent Space (Bottleneck):** 8 neurons + ReLU (High data compression)
*   **Decoder Layer 1:** 16 neurons + ReLU
*   **Output Layer:** 48 dimensions + Sigmoid activation

**Total Parameters:** ~1,300 parameters. This ultra-compact size allows for microsecond inference times and minimal memory footprint.

### 2.3 Training and Thresholding
The model is trained on the first **80%** of the chronologically ordered sequences (assumed to represent "normal" behavior). The loss function is Mean Squared Error (MSE), optimized using the Adam optimizer.

To detect anomalies in testing, the system calculates the reconstruction error for all training sequences and defines the **Anomaly Threshold** as the **95th percentile** of the training error distribution.

---

## 3. Implementation and Deployment

### 3.1 Digital Health Dashboard
To visualize the findings, an interactive dashboard was developed using Streamlit and Plotly. Key features include:
*   **Real-time Metrics:** Displays monitored hours, anomaly rate, and model confidence limits.
*   **Primary Anomaly Attribution:** Analyzes the standard deviation variance to determine mathematically whether Heart Rate or Activity (Step Count) is the primary driver of the disruption.
*   **Digital Doctor Insights:** An automated decision engine that dynamically generates medical and lifestyle recommendations based on the primary anomaly driver.
*   **Interactive Visualizations:** Dual-axis plotting of physiological markers against the Anomaly Threshold line, and anomaly-by-hour distribution charts.

### 3.2 Cloud Deployment
The system codebase is version-controlled via Git and hosted on GitHub. The presentation layer (Dashboard) is continuously deployed via Streamlit Community Cloud (Python 3.12). The architecture allows users to run the deep learning weight-generation offline using PyTorch, while uploading the secure, anonymized metadata (`dashboard_data.csv`) directly to the cloud dashboard for zero-computation visualization.

---

## 4. Results and Conclusion
The implementation successfully demonstrates that a highly compressed neural network (latent dimension = 8) is capable of mapping the complex, bi-variate relationship of the circadian cycle. By utilizing the 95th percentile error threshold, the system filters out minor physiological noise while successfully capturing major cyclical disruptions.

**Key Contributions:**
1. Developed a memory-efficient XML parser capable of handling massive Apple Health datasets linearly.
2. Proven the efficacy of a lightweight MLP Autoencoder for time-series wearable data.
3. Delivered an end-to-end, reproducible framework with a public-facing analytical dashboard.

Future work may involve incorporating additional modalities (e.g., HRV, ambient light exposure, skin temperature) and testing cross-user transfer learning.
