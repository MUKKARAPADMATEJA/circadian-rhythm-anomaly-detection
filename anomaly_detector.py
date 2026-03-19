import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os

# --- 1. Data Preprocessing ---
def prepare_data(csv_path):
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Convert dates to datetime
    df['start_date'] = pd.to_datetime(df['start_date'], utc=True)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    print("Resampling to hourly intervals...")
    # Pivot to get HeartRate and StepCount as separate columns
    df_pivot = df.pivot_table(index='start_date', columns='type', values='value', aggfunc='mean')
    
    # Resample to 1-hour intervals. Forward fill missing values (common for resting periods)
    # Then fill remaining NaNs with 0 (for steps) or mean (for HR)
    df_resampled = df_pivot.resample('1H').mean()
    
    if 'HeartRate' in df_resampled.columns:
        df_resampled['HeartRate'] = df_resampled['HeartRate'].ffill().bfill()
    else:
        df_resampled['HeartRate'] = 70.0 # fallback

    if 'StepCount' in df_resampled.columns:
        df_resampled['StepCount'] = df_resampled['StepCount'].fillna(0)
    else:
        df_resampled['StepCount'] = 0.0

    # Ensure no NaNs remain
    df_resampled = df_resampled.fillna(0)
    
    print(f"Total hourly records: {len(df_resampled)}")
    return df_resampled

def create_sequences(df, window_size=24):
    """
    Create sequences of window_size hours. 
    Each sequence represents a daily circadian rhythm cycle if window_size=24.
    """
    data = df.values
    sequences = []
    for i in range(len(data) - window_size + 1):
        seq = data[i:i + window_size]
        sequences.append(seq)
    
    return np.array(sequences)


# --- 2. Lightweight Autoencoder Model ---
class LightweightAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LightweightAutoencoder, self).__init__()
        # Compressed Representation
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
            nn.Sigmoid() # Output scaled between 0 and 1 (from MinMaxScaler)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# --- 3. Training and Evaluation ---
def train_and_detect(df, window_size=24, epochs=30):
    print("Normalizing data...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    # Create sequences
    sequences = []
    data_scaled = df_scaled.values
    for i in range(len(data_scaled) - window_size + 1):
        seq = data_scaled[i:i + window_size]
        sequences.append(seq)
    sequences = np.array(sequences)
    
    if len(sequences) == 0:
        print("Not enough data to create sequences. Need at least 24 hours of data.")
        return
        
    print(f"Created {len(sequences)} sequences of length {window_size} ({window_size} hours of data).")
    
    # Split: First 80% assumed normal for training, last 20% for detection
    split_idx = int(0.8 * len(sequences))
    train_seq = sequences[:split_idx]
    
    # Flatten the sequence for MLP Autoencoder: Shape [batch, window_size*features]
    num_features = train_seq.shape[2]
    input_dim = window_size * num_features
    
    X_train = torch.tensor(train_seq.reshape(-1, input_dim), dtype=torch.float32)
    
    print("Building model...")
    model = LightweightAutoencoder(input_dim=input_dim, hidden_dim=8)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Training the Lightweight Autoencoder...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, X_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    # Evaluation (Calculating Reconstruction Errors)
    print("Calculating expected normal error threshold...")
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train)
        train_errors = torch.mean((train_pred - X_train) ** 2, dim=1).numpy()
        
    # Set the anomaly threshold to the 95th percentile of the training errors
    threshold = float(np.percentile(train_errors, 95))
    print(f"Anomaly threshold (95th percentile): {threshold:.4f}")
    
    print("Running detection on entire dataset...")
    # Predict over all sequences to visualize in the dashboard
    X_all = torch.tensor(sequences.reshape(-1, input_dim), dtype=torch.float32)
    with torch.no_grad():
        all_pred = model(X_all)
        all_errors = torch.mean((all_pred - X_all) ** 2, dim=1).numpy()
        
    # Create DataFrame of results to match original time series
    # A sequence maps to the start time of that window
    results_dates = df.index[:len(all_errors)]
    
    results_df = pd.DataFrame({
        'timestamp': results_dates,
        'reconstruction_error': all_errors,
        'is_anomaly': all_errors > threshold
    })
    
    results_df.to_csv("anomaly_results.csv", index=False)
    
    # Save base data and threshold for dashboard
    df_raw = df.reset_index().rename(columns={'start_date': 'timestamp'})
    # Join with results
    df_combined = pd.merge(df_raw, results_df, on='timestamp', how='left')
    df_combined['threshold'] = threshold
    
    df_combined.to_csv("dashboard_data.csv", index=False)
    print("Saved 'dashboard_data.csv' for Streamlit Visualization.")
    
    # Save the model
    torch.save(model.state_dict(), "lightweight_autoencoder.pth")
    print("Model saved as 'lightweight_autoencoder.pth'")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "health_data_parsed.csv")
    
    if not os.path.exists(data_path):
        print(f"Could not find {data_path}")
    else:
        df = prepare_data(data_path)
        train_and_detect(df, window_size=24, epochs=50)
