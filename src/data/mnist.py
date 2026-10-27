# MNIST dataset loader

from pathlib import Path
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ermittelt Projektordner aus dem Dateispeicherort
DATA_DIR = PROJECT_ROOT / "data"

# ===== MNIST LADEN =====
def load_mnist(train=True):
    return datasets.MNIST(
        root=DATA_DIR,  # speichert Datensatz im Ordner data
        train=train,
        download=True,  # lädt fehlende Datensatzdateien herunter
        transform=transforms.ToTensor(),
    )

# ===== TEST =====
if __name__ == "__main__":
    dataset = load_mnist()

    image, label = dataset[0]   # lädt erstes Bild

    print("Anzahl Bilder: ", len(dataset))
    print("Bildform: ", image.shape)
    print("Richtige Ziffer: ", label)
    print("Wertebereich: ", image.min().item(), image.max().item())