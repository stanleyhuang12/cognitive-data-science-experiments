
from tools_rec import * 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def plot_linear_training_validation_loss(X, y, intervals): 
    r2_scores_tracker = []
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    steps = len(X_train) / intervals
    data_intervals = np.arange(steps, len(X_train), steps)
    data_intervals = list(data_intervals) + [data_intervals[-1] + steps]
    data_intervals = [int(x) for x in data_intervals]
    
    
    for i in range(len(data_intervals)): 
        linear_model = LinearRegression()
        X_train_subset = X_train[:int(data_intervals[i])]
        y_train_subset = y_train[:int(data_intervals[i])]
        
        linear_model.fit(X_train_subset, y_train_subset)
        y_pred = linear_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_scores_tracker.append(r2)
        print(f"R2 score for training size {int(data_intervals[i])}: {r2}")

    plt.figure(figsize=(10, 6))
    plt.plot(data_intervals, r2_scores_tracker, marker='o')
    plt.xlabel('Training Size (every 10% of training data)')
    plt.ylabel('R2 Score')
    plt.title('R2 Score vs Training Size for Linear Regression Model')
    plt.grid(True)
    plt.show()
    

def plot_nn_training_validation_loss(X, y, intervals, configs):
    # Split fixed train/test once
    X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, test_size=0.1, random_state=17)

    # Scale once on full training set
    scaler = StandardScaler().fit(X_train_full)
    X_train_full = scaler.transform(X_train_full)
    X_val = scaler.transform(X_val)

    n_total = len(X_train_full)
    steps = n_total // intervals
    data_intervals = np.arange(steps, n_total + 1, steps)

    r2_scores = []

    for n_samples in data_intervals:
        # Subset the training data
        X_sub = X_train_full[:n_samples]
        y_sub = y_train_full[:n_samples]

        # Convert to tensors
        X_sub_tensor = torch.from_numpy(X_sub).float()
        y_sub_tensor = torch.from_numpy(y_sub).float().view(-1, 1)
        X_val_tensor = torch.from_numpy(X_val).float()

        # Model + optimizer fresh every run
        model = initialize_model(configs)
        optimizer = optim.Adam(model.parameters(), lr=configs["lr"])
        criterion = nn.MSELoss()

        # Train loop
        dataset = TensorDataset(X_sub_tensor, y_sub_tensor)
        dataloader = DataLoader(dataset, batch_size=configs["batch_size"], shuffle=True)

        model.train()
        for epoch in range(40):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_tensor).numpy()
        r2 = r2_score(y_val, y_val_pred)
        r2_scores.append(r2)
        print(f"Training size: {n_samples}, R²: {r2:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(data_intervals, r2_scores, marker='o')
    plt.xlabel('Training Size')
    plt.ylabel('R² Score on Validation Set')
    plt.title('Neural Network: R² vs Training Size')
    plt.grid(True)
    plt.show()