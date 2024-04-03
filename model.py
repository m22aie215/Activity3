import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Set the device to GPU if available, else CPU
compute_device = "cuda" if torch.cuda.is_available() else "cpu"


# Function to plot training loss and accuracy curves
def display_training_metrics(loss_history, accuracy_history, opt_name):
    plt.plot(range(1, 11), loss_history, label="Train Loss")
    plt.plot(range(1, 11), accuracy_history, label="Train Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title(f"Training Metrics with {opt_name} Optimizer")
    plt.legend()
    plt.show()


# Data transformation and loading
data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
training_dataset = torchvision.datasets.STL10(
    root="./data", split="train", download=True, transform=data_transform
)
training_loader = torch.utils.data.DataLoader(
    training_dataset, batch_size=32, shuffle=True
)

# Model initialization
#version - 2 using densenet model
model = torchvision.models.densenet201(pretrained=True)

# List of optimizers
# version 2 using all others except Adam.
optimizer_list = [
    optim.Adagrad(model.fc.parameters()),
    optim.Adadelta(model.fc.parameters()),
    optim.RMSprop(model.fc.parameters()),
]

# Loss function
loss_function = nn.CrossEntropyLoss()

# Training loop
for opt in optimizer_list:
    loss_hist = []
    acc_hist = []

    opt_name = opt.__class__.__name__
    print(f"Training using {opt_name} optimizer...")

    for epoch in range(10):
        total_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for data, targets in training_loader:
            data, targets = data.to(compute_device), targets.to(compute_device)

            opt.zero_grad()
            predictions = model(data)
            loss = loss_function(predictions, targets)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            _, preds = predictions.max(1)
            total_preds += targets.size(0)
            correct_preds += preds.eq(targets).sum().item()

        epoch_loss = total_loss / len(training_loader)
        epoch_acc = 100.0 * correct_preds / total_preds

        loss_hist.append(epoch_loss)
        acc_hist.append(epoch_acc)

        print(
            f"Epoch [{epoch+1}/10], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
        )

    top_5_acc = np.sort(acc_hist)[-5:]
    print("---------------------------------------------------------")
    print("Top 5 Training Accuracies:", top_5_acc)
    print("---------------------------------------------------------")
    display_training_metrics(loss_hist, acc_hist, opt_name)
