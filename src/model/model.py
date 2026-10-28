# Convolutional Neural Network for recognising MNIST handwritten digits

# === Neural Network-Imports ===
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

# === Sonstige Imports ===
from pathlib import Path

# === Lokale Imports ===
from src.data.mnist import load_mnist

# ===== CNN-HAUPTKLASSE =====
class MNIST_CNN(nn.Module):     # vererbt nn-Module für PyTorch-Funktionen von Modellen
    def __init__(self):
        super().__init__()      # initialisiert ebendiese Basisklasse

        # === 1. CNN-Faltungsschicht/-Block ===
        self.conv1 = nn.Conv2d(
            in_channels=1,      # Eingabe-Bilder haben nur einen Eingabekanal: Graustufenwert
            out_channels=8,     # Wir arbeiten mit acht lernbaren Filtern
            kernel_size=3,      # Jeder Filter betrachtet einen Bereich von 3 x 3 Werten
            padding=1,          # Padding von 1 px um Eingabebild herum, damit 3 x 3-Filtermaske auch auf die Randpixel angewendet werden kann (also die Mitte des Filters auf einem Randpixel sein kann)
        )

        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # === 2. CNN-Faltungsschicht ===
        # nimmt Merkmalskarten der Ausgabe des ersten Blocks entgegen und verarbeitet diese
        self.conv2 = nn.Conv2d(
            in_channels=8,      # entsprechend der Ausgabe von Block 1
            out_channels=16,    # Ausgabe: 16 (neue) Merkmalskarten
            kernel_size=3,      # analog zu Block 1
            padding=1,
        )

        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # === Flattening und Klassifikation ===
        self.flatten = nn.Flatten(start_dim=1)  # ordnet die berechneten Merkmalswerte jedes Bildes zu einem langen Vektor an: [4, 16, 7, 7] -> [4, 784]

        self.classifier = nn.Linear(
            in_features=16 * 7 * 7,     # Eingabe vom Flattening
            out_features=10,            # 10 Ziffern
        )

    def forward(self, x):
        # === 1. Block ===
        # Faltung - Filterkerne werden über das Eingabebild geschoben
        x = self.conv1(x)       # [N, 1, 28, 28] -> [N, 8, 28, 28] (je ein Ausgabekanal pro Filter)

        # ReLU auf Ausgabe -> keine negativen Werte
        x = self.relu1(x)       # [N, 8, 28, 28] -> [N, 8, 28, 28]

        # (Max-)Pooling - reduziert Datenmenge (und damit den Aufwand), indem jede Merkmalskarte (= Ausgabekanal) in viele 2 x 2-Fenster geteilt und die größten Werte übernommen werden (sodass aus 28 x 28 ein 14 x 14-Fenster wird) 
        x = self.pool1(x)       # [N, 8, 28, 28] -> [N, 8, 14, 14]


        # === 2. Block ===
        x = self.conv2(x)       # [N, 1, 14, 14] -> [N, 16, 14, 14]
        x = self.relu2(x)       # [N, 1, 14, 14] -> [N, 16, 14, 14]
        x = self.pool2(x)       # [N, 1, 14, 14] -> [N, 16, 7, 7]

        # === Klassifikation ===
        x = self.flatten(x)     # [N, 16, 7, 7] -> [N, 784]
        x = self.classifier(x)  # [N, 784] -> [N, 10]

        return x
    
# ===== TRAINING =====
def train_model(model, train_data, device, epochs=5):
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True) # Batches mit max. 64 Bildern, Daten mischen
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # der Optimizer aktualisiert die Gewichte des Netzes

    model.train() # setzt Modell in Trainingsmodus (setzt training flag auf True)
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # löscht Gradienten des vorherigen Batches
            prediction = model(images) # Vorhersage machen
            loss = loss_fn(prediction, labels) # berechnet loss aus Vorhersage und wahren Antworten
            loss.backward() # berechnet für jedes Gewicht den Gradienten des Loss
            optimizer.step() # aktualisiert alle Gewichte anhand dieser Gradienten
        print(f"Epoche {epoch+1}: letzter Loss {loss.item():.4f}")


# ===== TEST =====
def evaluate(model, dataset, device):
    data_loader = DataLoader(dataset, batch_size=1000)

    model.eval() # setzt training flag auf False
    correct = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            prediction = model(images) # Vorhersage, hat Form [1000, 10], 1000 wegen batch_size und 10 sind die 10 Werte der Output-Neuronen
            # jeweils pro Zeile (mit den 10 Einträgen) wird immer der Index des Eintrags mit der höchsten Zahl gespeichert
            # Dieser Index sollte jetzt gleich der Zahl sein, welche auf dem Bild abgebildet ist
            predicted_numbers = prediction.argmax(dim=1)

            correct += (predicted_numbers == labels).sum().item() #sum liefert tensor mit einzigem Wert, .item() macht daraus einen integer
    return correct / len(dataset)


# ===== AUSFÜHRUNG =====
if __name__ == "__main__":
    torch.manual_seed(42)   # Initialisierung des Zufallsgenerators, damit die Ergebnisse reproduzierbar sind

    # === Gerät auswählen ===
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Verwendetes Gerät:", device)

    # === Trainings- und Validierungsdaten laden ===
    full_training_data = load_mnist(train=True)

    train_data, validation_data = random_split(
        full_training_data,
        [50_000, 10_000],
        generator=torch.Generator().manual_seed(42),
    )

    print("Trainingsbilder:", len(train_data))
    print("Validierungsbilder:", len(validation_data))

    # === Modell initialisieren und trainieren ===
    model = MNIST_CNN().to(device)

    train_model(model, train_data, device, epochs=5)

    accuracy = evaluate(model, validation_data, device)
    print(f"Validierungsgenauigkeit: {accuracy:.2%}")


    # === Trainiertes Modell sichern ===
    checkpoint_dir = (
        Path(__file__).resolve().parents[2] / "checkpoints"
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "mnist_baseline.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "validation_accuracy": accuracy,
        },
        checkpoint_path,
    )

    print("Referenzmodell gespeichert:", checkpoint_path)


    # === Gespeichertes Modell wiederherstellen und testen ===
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    loaded_model = MNIST_CNN().to(device)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])

    loaded_accuracy = evaluate(
        loaded_model,
        validation_data,
        device,
    )

    print(f"Genauigkeit nach dem Laden: {loaded_accuracy:.2%}")