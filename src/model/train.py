# Modellauswahl

# === Neural Network-Imports ===
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

# === Sonstige Imports ===
import argparse
from pathlib import Path

# === Lokale Imports ===
from src.data.mnist import load_mnist
from src.model.mlp import MNIST_MLP
from src.model.cnn import MNIST_CNN


# ===== TRAINING =====
def train_model(model, train_data, device, epochs=5):
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True) # Batches mit max. 64 Bildern, Daten mischen
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # der Optimizer aktualisiert die Gewichte des Netzes

    model.train() # setzt Modell in Trainingsmodus (setzt training flag auf True)
    for epoch in range(epochs):
        epoch_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()                       # löscht Gradienten des vorherigen Batches
            prediction = model(images)                  # Vorhersage machen
            loss = loss_fn(prediction, labels)          # berechnet Loss aus Vorhersage und wahren Antworten

            epoch_loss += loss.item() * labels.size(0)  # Durchschnitt von loss.item() wieder in Summe der Einzelverluste umrechnen -> Loss pro Epoche vergleichbar

            loss.backward()                             # berechnet für jedes Gewicht den Gradienten des Loss
            optimizer.step()                            # aktualisiert alle Gewichte anhand dieser Gradienten
            mean_loss = epoch_loss / len(train_data)
        print(f"Epoche {epoch + 1}: mittlerer Loss {mean_loss:.4f}")


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
def main():
    models = {"cnn": MNIST_CNN, "mlp": MNIST_MLP}   # vorhandene Modelle

    # === Fügt Parser hinzu, sodass beim Starten des Programms mit dem Switch --model cnn/mlp das entsprechende Modell trainiert wird ===
    parser = argparse.ArgumentParser(description="Trainiert ein Modell auf MNIST.")
    parser.add_argument("--model", choices=tuple(models), default="cnn")
    args = parser.parse_args()

    model_class = models[args.model]

    torch.manual_seed(42)   # Initialisierung des Zufallsgenerators, damit die Ergebnisse reproduzierbar sind

    # === Gerät auswählen ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modell: {args.model} | Gerät: {device}")

    # === Trainings- und Validierungsdaten laden ===
    full_training_data = load_mnist(train=True)
    train_data, validation_data = random_split(
            full_training_data,
            [50_000, 10_000],
            generator=torch.Generator().manual_seed(42),
    )

    # === Modell initialisieren und trainieren ===
    model = model_class().to(device)
    train_model(model, train_data, device=device, epochs=5)

    accuracy = evaluate(model, validation_data, device)
    print(f"Validierungsgenauigkeit: {accuracy:.2%}")

    # === Trainiertes Modell sichern ===
    checkpoint_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = "mnist_baseline.pt" if args.model == "cnn" else "mnist_mlp_baseline.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "validation_accuracy": accuracy,
        },
        checkpoint_path,
    )
    print(f"Referenzmodell gespeichert: {checkpoint_path}")

    # === Gespeichertes Modell wiederherstellen und testen ===
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    loaded_model = model_class().to(device)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])

    loaded_accuracy = evaluate(loaded_model, validation_data, device)
    print(f"Genauigkeit nach dem Laden: {loaded_accuracy:.2%}")

if __name__ == "__main__":
    main()