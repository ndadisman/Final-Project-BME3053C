import numpy as np
import pyedflib
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

file_path = "C:\\Users\\nldad\\Documents\\Final-Project-BME3053C\\data"

def load_edf(file_path):
    edf = pyedflib.EdfReader(file_path)
    signal_data = edf.readSignal(0)  # Assuming you are using the first channel
    return signal_data

def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=100):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(1, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def extract_features(filtered_data):
    features = []
    features.append(np.mean(filtered_data))
    features.append(np.std(filtered_data))
    features.append(np.ptp(filtered_data))  # Peak-to-peak amplitude
    # Add more features if necessary (e.g., FFT-based)
    return np.array(features)

def train_model(X, y):
    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Using cross-validation to estimate model performance
    model = RandomForestClassifier(n_estimators=100)
    cross_val_results = cross_val_score(model, X_scaled, y, cv=5)
    print(f'Cross-validated accuracy: {np.mean(cross_val_results):.4f}')
    
    model.fit(X_scaled, y)  # Fit the model on t
