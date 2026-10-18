
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "handwritten" / "images"
PROCESSED_DIR = BASE_DIR / "handwritten" / "processed"
MODEL_PATH = BASE_DIR / "models" / "experiment_2_best_mlp.pth"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


model = MLP().to(device)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()


def prepare_image(image_path):
    image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image)
    gray = np.array(image)

    border = np.concatenate([
        gray[0, :], gray[-1, :],
        gray[:, 0], gray[:, -1]
    ])

    # Açık arka planı ters çevir.
    if np.median(border) > 127:
        gray = 255 - gray

    mask = gray > 40

    if not mask.any():
        raise ValueError("Görüntüde rakam bulunamadı.")

    ys, xs = np.where(mask)
    cropped = gray[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    digit = Image.fromarray(cropped.astype(np.uint8))
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(
        digit,
        ((28 - digit.width) // 2, (28 - digit.height) // 2)
    )

    array = np.array(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)

    return canvas, tensor.to(device)


if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Klasör bulunamadı: {IMAGE_DIR}")

# digit_0, digit_1, ... digit_9 dosyalarını bul.
image_files = {}

for path in IMAGE_DIR.iterdir():
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        continue

    match = re.fullmatch(r"digit_(\d+)", path.stem)
    if match:
        digit_number = int(match.group(1))
        if 0 <= digit_number <= 9:
            image_files[digit_number] = path

if not image_files:
    raise FileNotFoundError(
        "digit_0.jpeg ile digit_9.jpeg arasındaki dosyaları "
        "handwritten/images klasörüne ekle."
    )

results = []

with torch.no_grad():
    for expected_digit in sorted(image_files):
        image_path = image_files[expected_digit]

        processed_image, tensor = prepare_image(image_path)

        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)

        prediction = int(logits.argmax(dim=1).item())
        confidence = float(
            probabilities[0, prediction].item() * 100
        )

        correct = prediction == expected_digit

        processed_image.save(
            PROCESSED_DIR / f"digit_{expected_digit}_28x28.png"
        )

        results.append({
            "expected": expected_digit,
            "prediction": prediction,
            "confidence": confidence,
            "correct": correct,
            "image": processed_image
        })


print("\n--- 0-9 EL YAZISI RAKAM TANIMA SONUÇLARI ---")

for result in results:
    status = "DOĞRU" if result["correct"] else "YANLIŞ"

    print(
        f"Beklenen: {result['expected']} | "
        f"Tahmin: {result['prediction']} | "
        f"Güven: %{result['confidence']:.2f} | "
        f"{status}"
    )

correct_count = sum(result["correct"] for result in results)
total_count = len(results)
accuracy = correct_count / total_count * 100

print(f"\nDoğru tahmin: {correct_count}/{total_count}")
print(f"El yazısı test doğruluğu: %{accuracy:.2f}")

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()

for ax in axes:
    ax.axis("off")

for ax, result in zip(axes, results):
    ax.imshow(result["image"], cmap="gray", vmin=0, vmax=255)
    ax.set_title(
        f"Beklenen: {result['expected']}\n"
        f"Tahmin: {result['prediction']}\n"
        f"%{result['confidence']:.1f}",
        color="green" if result["correct"] else "red"
    )
    ax.axis("off")

plt.suptitle("0-9 El Yazısı Rakam Tanıma")
plt.tight_layout()
plt.show()
