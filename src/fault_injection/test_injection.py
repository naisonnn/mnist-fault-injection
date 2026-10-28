# First injection test for the MNIST CNN model

from pathlib import Path

import torch
from torch import nn
from pytorchfi.core import fault_injection

from src.model.model import MNIST_CNN
from src.data.mnist import load_mnist

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # === Modell aus gespeichertem Checkpoint laden ===
    project_root = Path(__file__).resolve().parents[2]
    checkpoint_path = (
        project_root / "checkpoints" / "mnist_baseline.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = MNIST_CNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # === Fehler-Injektion ===
    injector = fault_injection(
        model,                              # Das geladene Modell
        batch_size=1,                       # Einzelnes Bild
        input_shape=[1, 28, 28],            # Keine Batchdimension -> Graustufenbild mit 28 x 28 px
        layer_types=[nn.Conv2d],            # Faltungsschichten als möglche Injektionspunkte auswählens
        use_cuda=device.type == "cuda",     # Zum Modell passendes Gerät wählen  
    )

    injector.print_pytorchfi_layer_summary()

    # Schritt 1: Einzelnes Bild vorbereiten
    dataset = load_mnist(train=True)
    image, label = dataset[0]
    images = image.unsqueeze(0).to(device)

    # Schritt 2: Injektion festlegen
    faulty_model = injector.declare_neuron_fi(
        layer_num=[0],  # conv1
        batch=[0],      # erstes Bild
        dim1=[0],       # erster Ausgabekanal
        dim2=[14],      # Zeile
        dim3=[14],      # Spalte
        value=[1000.0]  # neuer Aktivierungswert
    )

    # Schritt 3: Aktivierungen beobachten
    activations = []

    def record_activation(module, inputs, output):
        activations.append(output.detach().clone())

    reference_handle = model.conv1.register_forward_hook(
        record_activation
    )

    faulty_handle = faulty_model.conv1.register_forward_hook(
        record_activation
    )

    # Schritt 4: Beide Modelle ausführen
    try:
        with torch.no_grad():
            reference_logits = model(images)

            injector.reset_current_layer()
            faulty_logits = faulty_model(images)

    finally:
        reference_handle.remove()
        faulty_handle.remove()


    # Schritt 5: Ergebnisse vergleichen
    reference_activation, faulty_activation = activations
    position = (0, 0, 14, 14)

    print("Wert vorher:", reference_activation[position].item())
    print("Wert nachher:", faulty_activation[position].item())

    changed = (
        reference_activation != faulty_activation
    ).sum().item()

    print("Geänderte Aktivierungswerte:", changed)
    print("Richtige Ziffer:", label)
    print("Referenzvorhersage:", reference_logits.argmax(dim=1).item())
    print("Vorhersage mit Fehler:", faulty_logits.argmax(dim=1).item())

if __name__ == "__main__":
    main()