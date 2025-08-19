# optimized_train_model_simple.py
import gc
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
import glob

DATA_DIR = Path("data")
BATCH_SIZE = 1024
EPOCHS = 5
INPUT_DIM = 30 + 10  # user + restaurant features
LEARNING_RATE = 1e-6  # Optimized learning rate
MAX_TIME_PER_EPOCH = 45  # seconds


# DO NOT EDIT MODEL ARCHITECTURE
class RankNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)


def process_one_file(file_path, user_features, restaurant_features):
    """Load one file and merge with features"""
    # Load training data
    train_data = pd.read_parquet(file_path)
    
    # Merge with user and restaurant features
    merged_data = train_data.merge(user_features, on='user_id', how='left')
    merged_data = merged_data.merge(restaurant_features, on='restaurant_id', how='left')
    
    # Extract features (all columns starting with 'f')
    feature_cols = [col for col in merged_data.columns if col.startswith('f')]
    X = merged_data[feature_cols].values.astype(np.float32)
    y = merged_data['click'].values.astype(np.float32)
    
    # Clean up memory
    del train_data, merged_data
    gc.collect()
    
    return X, y


def train_on_data(model, criterion, optimizer, X, y, max_time):
    """Train model on the provided data"""
    start_time = time.time()
    num_samples = len(X)
    
    if num_samples == 0:
        return 0.0, 0
    
    # Shuffle the data
    indices = np.random.permutation(num_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    total_loss = 0.0
    batches_processed = 0
    
    # Train in batches
    for start_idx in range(0, num_samples, BATCH_SIZE):
        # Check time limit
        if time.time() - start_time > max_time:
            break
            
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        
        # Create tensors
        batch_X = torch.from_numpy(X_shuffled[start_idx:end_idx])
        batch_y = torch.from_numpy(y_shuffled[start_idx:end_idx])
        
        # Training step
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batches_processed += 1
    
    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0
    return avg_loss, batches_processed


def train():
    # Load user and restaurant features once
    user_features = pd.read_parquet(DATA_DIR / "user_features.parquet")
    restaurant_features = pd.read_parquet(DATA_DIR / "restaurant_features.parquet")
    
    # Get all training files
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.parquet")))
    
    # Initialize model
    model = RankNet(INPUT_DIM)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        epoch_loss = 0.0
        total_batches = 0
        total_samples = 0
        files_processed = 0
        
        # Shuffle file order each epoch for better learning
        shuffled_files = np.random.permutation(train_files)
        
        # Process files one by one
        for file_idx, file_path in enumerate(shuffled_files):
            # Check time limit
            elapsed_time = time.time() - epoch_start
            if elapsed_time > MAX_TIME_PER_EPOCH - 3:
                break
            
            # Load and process one file
            X, y = process_one_file(file_path, user_features, restaurant_features)
            total_samples += len(X)
            
            # Train on this file's data
            remaining_time = MAX_TIME_PER_EPOCH - (time.time() - epoch_start)
            file_loss, file_batches = train_on_data(model, criterion, optimizer, X, y, remaining_time)
            
            if file_batches > 0:
                epoch_loss += file_loss * file_batches
                total_batches += file_batches
            
            files_processed += 1
            
            # Clean up memory
            del X, y
            gc.collect()
    
    # Save model
    model_scripted = torch.jit.script(model)
    model_scripted.save(DATA_DIR / "model.pt")


if __name__ == "__main__":
    train()