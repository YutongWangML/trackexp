"""
Example of using trackexp to track a neural network training process for IRIS classification.

This example demonstrates:
1. Tracking training and validation metrics
2. Saving model checkpoints
3. Storing hyperparameters as metadata
4. Plotting and saving learning curves
5. Classification with 3 classes using PyTorch
"""

import sys
import os
import pickle
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Add the parent directory to the path so we can import trackexp
# This is only needed when running the example without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import trackexp

class IrisNN(nn.Module):
    """A simple neural network implementation for IRIS classification."""

    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the network."""
        super(IrisNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass."""
        # Hidden layer
        x = self.layer1(x)
        x = self.relu(x)

        # Output layer (logits)
        x = self.layer2(x)

        return x

def save_model(context, name, identifier, model):
    """Save model checkpoint."""
    filepath = f"model_checkpoint_{identifier}.pth"
    torch.save(model.state_dict(), filepath)
    return filepath

def save_plot(context, name, identifier, data):
    """Save a plot of the learning curve."""
    filepath = f"learning_curve_{identifier}.png"

    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(f"Learning Curve (Epoch {identifier})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath

def load_iris_data():
    """Load and prepare the IRIS dataset."""
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    return X_tensor, y_tensor

def main():
    """Run the neural network example."""
    # Initialize experiment tracking
    trackexp.init("iris_classification")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Load IRIS data
    print("Loading IRIS dataset...")
    X, y = load_iris_data()

    # Split into train, validation, and test sets
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    test_size = len(X) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        TensorDataset(X, y), [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Define hyperparameters
    input_size = 4  # IRIS has 4 features
    hidden_size = 10
    output_size = 3  # 3 classes in IRIS
    learning_rate = 0.01
    epochs = 100

    # Save hyperparameters as metadata
    config = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "data_size": len(X),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size
    }
    trackexp.metadata(config)
    print("Experiment config saved as metadata.")

    # Initialize the model
    model = IrisNN(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model and track progress
    print("Training model...")
    wallclocktime = 0
    train_losses = []

    for epoch in range(epochs):
        # Start the timer for this epoch
        epoch_start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        all_logits = []

        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            all_logits.append(logits.detach().cpu().numpy())

        # Calculate epoch loss
        epoch_loss = running_loss / train_size
        train_losses.append(epoch_loss)

        # Stop the timer for this epoch
        epoch_end_time = time.time()
        wallclocktime += (epoch_end_time - epoch_start_time)

        # Log training metrics
        trackexp.log("training", "loss", epoch, epoch_loss)

        # Log the final batch logits
        trackexp.log("training", "logits", epoch, np.vstack(all_logits))

        # Log the wallclock time
        trackexp.log("training", "wallclocktime", epoch, wallclocktime)

        # Evaluate on validation set
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss = val_loss / val_size
            val_accuracy = correct / total

            # Log validation metrics
            trackexp.log("validation", "loss", epoch, val_loss)
            trackexp.log("validation", "accuracy", epoch, val_accuracy)

            trackexp.log("validation", "wallclocktime", epoch, wallclocktime)
            # Save model checkpoint
            trackexp.log("checkpoints", "model", epoch, model, savefunc=save_model)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / test_size
    test_accuracy = correct / total

    # Log test metrics
    trackexp.log("metrics", "test_loss", "final", test_loss)
    trackexp.log("metrics", "test_accuracy", "final", test_accuracy)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Get current experiment info to show where data is stored
    exp_info = trackexp.get_current_experiment()
    print(f"\nExperiment data stored in: {exp_info['path']}")

if __name__ == "__main__":
    main()
