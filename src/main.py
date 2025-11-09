
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    data = pd.read_csv(filepath)
    print(f"Data loaded successfully with shape: {data.shape}")
    return data

def preprocess_data(data):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data.drop("Class", axis=1))
    return X_scaled, data["Class"]

def train_isolation_forest(X_scaled):
    model = IsolationForest(n_estimators=100, contamination=0.0016, random_state=42)
    model.fit(X_scaled)
    return model

if __name__ == "__main__":
    print("🚀 Running Credit Card Fraud Detection (Unsupervised Learning)...")
    # Example placeholder for dataset path
    # data = load_data('data/creditcard.csv')
    # X_scaled, y = preprocess_data(data)
    # model = train_isolation_forest(X_scaled)
    print("✅ Project setup complete. Add your dataset and start training!")
