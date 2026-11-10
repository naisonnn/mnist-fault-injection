# First injection test for CNN and MLP

# === Neural Network-Imports ===
import torch
from torch import nn
from pytorchfi.core import fault_injection

# === Sonstige Imports ===
import argparse
from pathlib import Path

# === Lokale Imports ===
from src.model.cnn import MNIST_CNN
from src.model.mlp import MNIST_MLP
from src.data.mnist import load_mnist

INJECTED_VALUE = 1000.0

# === Vorkonfigurationen: Modellklasse, Checkpoint-Dateiname, Schichttyp, Schichtname und Position im Tensor, in den injiziert wird ===
MODEL_CONFIGS = {
    "cnn": (MNIST_CNN, "mnist_baseline.pt", nn.Conv2d, "conv1", (0, 0, 14, 14)),
    "mlp": (MNIST_MLP, "mnist_mlp_baseline.pt", nn.Linear, "hidden", (0, 0)),
}

def run_injection(model_name, images, label, checkpoint_dir):
    model_class, checkpoint_name, layer_type, layer_name, position = MODEL_CONFIGS[model_name]
    device = images.device
    checkpoint_path = checkpoint_dir / checkpoint_name
    print(f"\n=== {model_name.upper()} | {layer_name} | Position {position} ===")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = model_class().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        injector = fault_injection(
            model,
            batch_size=1,
            input_shape=[1, 28, 28],
            layer_types=[layer_type],
            use_cuda=device.type == "cuda",
        )

    injector.print_pytorchfi_layer_summary()

    batch_index, feature_index = position[:2]
    row, column = position[2:] if len(position) == 4 else (None, None)
    faulty_model = injector.declare_neuron_fi(
        layer_num=[0],  # erste Schicht
        batch=[batch_index],
        dim1=[feature_index],
        dim2=[row],
        dim3=[column],
        value=[INJECTED_VALUE],
    )

    activations = []

    def record_activation(module, inputs, output):
        activations.append(output.detach().clone())

    # Beim Fehlermodell läuft der Beobachter nach dem Injektions-Hook
    reference_handle = getattr(model, layer_name).register_forward_hook(record_activation)
    faulty_handle = getattr(faulty_model, layer_name).register_forward_hook(record_activation)

    try:
        with torch.no_grad():
            reference_logits = model(images)
            injector.reset_current_layer()
            faulty_logits = faulty_model(images)
    finally:
        reference_handle.remove()
        faulty_handle.remove()
        injector.fi_reset()

    reference_activation, faulty_activation = activations

    changed_positions = (reference_activation != faulty_activation).nonzero()
    print("Tatsächlich geänderte Positionen:", changed_positions.tolist())

    before = reference_activation[position].item()
    after = faulty_activation[position].item()
    changed = (reference_activation != faulty_activation).sum().item()
    passed = after == INJECTED_VALUE and changed == 1

    # === Ausgabe ===
    print(f"Wert vorher: {before}")
    print(f"Wert nachher: {after}")
    print(f"Geänderte Aktivierungswerte: {changed}")
    print(f"Richtige Ziffer: {label}")
    print(f"Referenzvorhersage: {reference_logits.argmax(dim=1).item()}")
    print(f"Vorhersage mit Fehler: {faulty_logits.argmax(dim=1).item()}")
    print("Injektionstest: " + ("bestanden" if passed else "fehlgeschlagen"))
    return passed

def main():
    parser = argparse.ArgumentParser(description="Testet eine Aktivierungsinjektion.")
    parser.add_argument("--model", choices=("both", "cnn", "mlp"), default="both")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    image, label = load_mnist(train=True)[0]
    images = image.unsqueeze(0).to(device)

    selected_models = MODEL_CONFIGS if args.model == "both" else (args.model,)
    results = [
        run_injection(name, images, label, checkpoint_dir) for name in selected_models
    ]

    return 0 if all(results) else 1

if __name__ == "__main__":
    raise SystemExit(main())