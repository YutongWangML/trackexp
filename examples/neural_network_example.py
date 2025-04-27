"""
Example of using trackexp to track a neural network training process for IRIS classification.

This example demonstrates:
1. Tracking training and validation metrics
2. Saving model checkpoints
3. Storing hyperparameters as metadata
4. Plotting and saving learning curves
5. Classification with 3 classes using PyTorch
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import trackexp

class SimpleNN(nn.Module):
    """A simple neural network implementation for IRIS classification."""

    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the network."""
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass."""
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


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
    trackexp.init("iris_classification", verbose=True)

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

    model = SimpleNN(input_size, hidden_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)


    training_start_time = time.time()
    for epoch in range(epochs):
        trackexp.start_timer("training", epoch)
        # Training phase
        model.train()
        running_loss = 0.0
        all_logits = []

        # Start the timer for this epoch

        minibatch = 0
        for inputs, labels in train_loader:
            # trackexp.start_timer("training_inner", (epoch,minibatch))
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

            # trackexp.log("training_inner", "loss", (epoch, minibatch), loss.item())
            # trackexp.stop_timer("training_inner", (epoch,minibatch))

            minibatch += 1


        trackexp.stop_timer("training", epoch)
        trackexp.print_log("training")

        epoch_loss = running_loss / train_size

        # Logging
        trackexp.log("training", "loss", epoch, epoch_loss)
        trackexp.log("training", "logits", epoch, np.vstack(all_logits))

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
            trackexp.print_log("validation")

    training_end_time = time.time()
    print(f"Total training time: {training_end_time - training_start_time}")
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
    trackexp.log("metrics", "test_loss", None, test_loss)
    trackexp.log("metrics", "test_accuracy", None, test_accuracy)
    trackexp.print_log("metrics")

    # Get current experiment info to show where data is stored
    exp_info = trackexp.get_current_experiment()
    print(f"\nExperiment data stored in: {exp_info['path']}")

if __name__ == "__main__":
    main()
