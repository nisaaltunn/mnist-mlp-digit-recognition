
import copy
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# 1. Genel ayarlar
SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Kullanılan cihaz:", DEVICE)

Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)


# 2. MNIST veri seti
transform = transforms.ToTensor()

full_train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [55000, 5000],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# 3. MLP mimarisi
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


model = MLP().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 4. Bir veri kümesindeki loss ve accuracy değerlerini hesaplama
def evaluate(data_loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


# 5. Eğitim döngüsü
history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": []
}

best_val_loss = float("inf")
best_model_state = None

start_time = time.perf_counter()

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    train_loss = total_loss / total
    train_accuracy = 100.0 * correct / total

    val_loss, val_accuracy = evaluate(val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_accuracy"].append(val_accuracy)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())


training_time = time.perf_counter() - start_time

# 6. En iyi doğrulama sonucuna ait ağırlıkları geri yükle
model.load_state_dict(best_model_state)

torch.save(model.state_dict(), "models/best_mlp.pth")


# 7. Test kümesini yalnızca final değerlendirmesinde kullan
test_loss, test_accuracy = evaluate(test_loader)

print("\n--- SONUÇLAR ---")
print(f"En iyi validation loss: {best_val_loss:.4f}")
print(f"Son epoch train loss: {history['train_loss'][-1]:.4f}")
print(f"Son epoch validation loss: {history['val_loss'][-1]:.4f}")
print(f"Son epoch train accuracy: {history['train_accuracy'][-1]:.2f}%")
print(f"Son epoch validation accuracy: {history['val_accuracy'][-1]:.2f}%")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.2f}%")
print(f"Eğitim süresi: {training_time:.2f} saniye")


# 8. Loss grafiği
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_loss"], label="Training Loss")
plt.plot(epochs_range, history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training ve Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/loss_curve.png", dpi=150)
plt.show()


# 9. Accuracy grafiği
plt.figure(figsize=(8, 5))
plt.plot(
    epochs_range,
    history["train_accuracy"],
    label="Training Accuracy"
)
plt.plot(
    epochs_range,
    history["val_accuracy"],
    label="Validation Accuracy"
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training ve Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/accuracy_curve.png", dpi=150)
plt.show()


# 10. Sonuçları dosyaya kaydet
results = {
    "model": "MLP",
    "hidden_layers": [128, 64],
    "activation": "ReLU",
    "loss_function": "CrossEntropyLoss",
    "optimizer": "Adam",
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "seed": SEED,
    "device": str(DEVICE),
    "train_samples": len(train_dataset),
    "validation_samples": len(val_dataset),
    "test_samples": len(test_dataset),
    "best_validation_loss": best_val_loss,
    "final_train_loss": history["train_loss"][-1],
    "final_validation_loss": history["val_loss"][-1],
    "final_train_accuracy": history["train_accuracy"][-1],
    "final_validation_accuracy": history["val_accuracy"][-1],
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
    "training_time_seconds": training_time,
    "history": history
}

with open("results/experiment_1.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

print("\nModel ve deney sonuçları kaydedildi.")
