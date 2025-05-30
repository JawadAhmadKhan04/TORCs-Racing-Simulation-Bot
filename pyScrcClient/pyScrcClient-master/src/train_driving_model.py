import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import os

# Define the neural network architecture
class DrivingModel(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(DrivingModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 4)  # 4 outputs: brake, accelerate, turn, gear
        )
        
    def forward(self, x):
        return self.network(x)

# Custom dataset class
class DrivingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_driving_model.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def main():
    # Load dataset
    csv_path = 'dataset.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return
    
    df = pd.read_csv(csv_path)
    
    # Select relevant features (inputs)
    feature_columns = [
        'angle', 'rpm', 'speedX', 'speedY', 'speedZ', 'trackPos',
        'closest_opponent_distance', 'closest_opponent_direction',
        'relative_opponent_speed', 'minimum_track_distance',
        'track_curvature', 'track_width', 'num_nearby_opponents',
        'car_acceleration', 'steering_rate'
    ] + [f'track_{i}' for i in range(19)]
    
    # Select targets (outputs)
    target_columns = ['brake', 'accel', 'steer', 'gear']
    
    # Drop rows with missing values
    df = df.dropna(subset=feature_columns + target_columns)
    
    X = df[feature_columns].astype(float)
    y = df[target_columns].astype(float)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, 'driving_scaler.joblib')
    
    # Create datasets and dataloaders
    train_dataset = DrivingDataset(X_train_scaled, y_train.values)
    val_dataset = DrivingDataset(X_test_scaled, y_test.values)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DrivingModel(input_size=len(feature_columns)).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=50, device=device
    )
    
    print("Training completed!")
    print("Model and scaler saved as 'best_driving_model.pth' and 'driving_scaler.joblib'")

if __name__ == "__main__":
    main() 