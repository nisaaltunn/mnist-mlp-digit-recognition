from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import label, find_objects


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
    raise FileNotFoundError(
        f"Model bulunamadi: {MODEL_PATH}\n"
        "Once experiment_2.py dosyasini calistir."
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)
model.load_state_dict(checkpoint)
model.eval()


def process_full_number_image(image_path):
    """Tek bir görüntü dosyasındaki tüm okul numarasını soldan sağa segmentlere ayırır."""
    image = Image.open(image_path).convert("L")
    gray = np.array(image)

    # Arka plan beyazsa görüntüyü ters çevir.
    border = np.concatenate([
        gray[0, :], gray[-1, :],
        gray[:, 0], gray[:, -1]
    ])
    if np.median(border) > 127:
        gray = 255 - gray

    # Otsu eşikleme ile gürültüleri tamamen yok et (saf siyah/beyaz yap).
    hist, bin_edges = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    valid = (weight1 > 0) & (weight2 > 0)
    
    if valid.any():
        mean1 = np.cumsum(hist * bin_mids) / np.maximum(weight1, 1)
        mean2 = (np.cumsum((hist * bin_mids)[::-1])[::-1]) / np.maximum(weight2, 1)
        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        thresh = bin_mids[np.argmax(variance)]
    else:
        thresh = 127

    # Rakam dışındaki her şeyi tamamen siyah (0) yap
    binary = np.where(gray > max(thresh, 40), gray, 0)
    mask = binary > 0

    if not mask.any():
        raise ValueError(f"{image_path.name} içinde rakam bulunamadı.")

    # Bağlı bileşenleri (Connected Components) bul
    labeled_array, num_features = label(mask)
    slices = find_objects(labeled_array)

    # Rakamları yatay konumlarına (soldan sağa) göre sırala
    digit_boxes = []
    for slc in slices:
        if slc is None:
            continue
        y_slc, x_slc = slc
        # Çok küçük gürültüleri ele
        if (y_slc.stop - y_slc.start < 5) or (x_slc.stop - x_slc.start < 3):
            continue
        digit_boxes.append((x_slc.start, slc))

    # Soldan sağa sıralama
    digit_boxes.sort(key=lambda x: x[0])

    digit_results = []
    with torch.no_grad():
        for idx, (_, slc) in enumerate(digit_boxes):
            y_slc, x_slc = slc
            cropped = binary[y_slc, x_slc]

            # En-boy oranını koruyarak 20x20 alana sığdır.
            digit_img = Image.fromarray(cropped.astype(np.uint8))
            digit_img.thumbnail((20, 20), Image.Resampling.LANCZOS)

            # MNIST benzeri saf siyah 28x28 zemin oluştur ve ortala.
            canvas = Image.new("L", (28, 28), 0)
            x_offset = (28 - digit_img.width) // 2
            y_offset = (28 - digit_img.height) // 2
            canvas.paste(digit_img, (x_offset, y_offset))

            # İşlenmiş tekil rakamı kaydet
            output_path = PROCESSED_DIR / f"digit_part_{idx+1}_28x28.png"
            canvas.save(output_path)

            # Normalizasyon ve tahmin
            arr = np.array(canvas, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            prediction = int(logits.argmax(dim=1).item())
            confidence = float(probs[0, prediction].item() * 100)

            digit_results.append({
                "digit_index": idx + 1,
                "prediction": prediction,
                "confidence": confidence,
                "image": canvas
            })

    return digit_results


# Görüntü dosyasını bul (Örn: adında 'all' veya 'num' geçen ya da klasördeki ilk dosya)
image_files = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
) if IMAGE_DIR.exists() else []

if not image_files:
    raise FileNotFoundError(
        "handwritten/images klasorune rakam PNG veya JPG "
        "dosyalari eklemelisin."
    )

# Eğer tam okul numarasını içeren tek bir dosya varsa (örn: all_digit.png) onu seç
target_file = next((f for f in image_files if "all" in f.stem.lower() or "num" in f.stem.lower()), image_files[0])
print(f"İşlenen tam numara görseli: {target_file.name}")

results = process_full_number_image(target_file)

if not results:
    raise RuntimeError("Hiçbir rakam segmentlere ayrılamadı.")

# --- OKUL NUMARASINI BİRLEŞTİRME VE YAZDIRMA ---
full_student_id = "".join([str(res["prediction"]) for res in results])

print("\n" + "="*40)
print(f"🎯 TAHMİN EDİLEN OKUL NUMARASI: {full_student_id}")
print("="*40 + "\n")

print("--- TEKİL RAKAM TAHMİN DETAYLARI ---")
for res in results:
    print(f"Rakam {res['digit_index']}: Tahmin = {res['prediction']} (%{res['confidence']:.2f} güven)")

# Tahminleri gorsellestir.
count = len(results)
fig, axes = plt.subplots(
    1, count, figsize=(2.5 * count, 3)
)
axes = np.atleast_1d(axes)

for ax, result in zip(axes, results):
    ax.imshow(result["image"], cmap="gray", vmin=0, vmax=255)
    ax.set_title(
        f"Rakam {result['digit_index']}\n"
        f"Tahmin: {result['prediction']}\n"
        f"%{result['confidence']:.1f}"
    )
    ax.axis("off")

plt.suptitle(f"Bulunan Okul Numarası: {full_student_id}", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()