# First injection test for CNN and MLP

# === Neural Network-Imports ===
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
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

def run_injection(model_name, dataset, device, checkpoint_dir):
    total = len(dataset)
    if total == 0:
        raise ValueError("Der Auswertungsdatensatz ist leer.")
    model_class, checkpoint_name, layer_type, layer_name, position = MODEL_CONFIGS[model_name]
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

    # batch_size=1: Die Injektion bei batch=[0] trifft jedes ausgewertete Bild.
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    correct_reference = correct_faulty = 0
    invalid_reference = invalid_faulty = 0
    valid_pairs = changed_predictions = new_errors = 0
    verified_injections = 0

    try:
        with torch.no_grad():
            for count, (images, labels) in enumerate(data_loader, start=1):
                images = images.to(device)
                label = labels.item()
                # Nur die Aktivierungen des aktuellen Bildes aufbewahren.
                activations.clear()

                reference_logits = model(images)
                injector.reset_current_layer()  # Vor JEDEM Fehlerlauf zurücksetzen.
                faulty_logits = faulty_model(images)

                reference_activation, faulty_activation = activations
                before = reference_activation[position].item()
                after = faulty_activation[position].item()
                changed = (reference_activation != faulty_activation).sum().item()
                verified_injections += int(after == INJECTED_VALUE and changed == 1)

                if count == 1:
                    print(f"Erstes Bild: Wert vorher {before}, nachher {after}")
                    print(f"Erstes Bild: geänderte Aktivierungswerte {changed}")

                # NaN/Inf dürfen nicht als gewöhnliche Vorhersage gezählt werden.
                reference_valid = torch.isfinite(reference_logits).all().item()
                faulty_valid = torch.isfinite(faulty_logits).all().item()
                reference_number = faulty_number = None
                if reference_valid:
                    reference_number = reference_logits.argmax(dim=1).item()
                    correct_reference += int(reference_number == label)
                else:
                    invalid_reference += 1

                if faulty_valid:
                    faulty_number = faulty_logits.argmax(dim=1).item()
                    correct_faulty += int(faulty_number == label)
                else:
                    invalid_faulty += 1

                if reference_valid and faulty_valid:
                    valid_pairs += 1
                    changed_predictions += int(reference_number != faulty_number)

                # Unter den sauberen, korrekt erkannten Bildern zählen ungültige
                # Fehlerläufe ebenfalls als neu entstandenes Erkennungsversagen.
                if reference_valid and reference_number == label:
                    new_errors += int(not faulty_valid or faulty_number != label)

                if count % 1000 == 0:
                    print(f"Ausgewertet: {count}/{total}")
    finally:
        reference_handle.remove()
        faulty_handle.remove()
        injector.fi_reset()

    # Ungültige Ausgaben zählen im Nenner der Gesamtgenauigkeit mit.
    accuracy = correct_reference / total
    faulty_accuracy = correct_faulty / total
    passed = verified_injections == total
    print(f"Datensatzgröße: {total}")
    print(f"Accuracy ohne Injektion: {accuracy:.2%}")
    print(f"Accuracy mit Injektion: {faulty_accuracy:.2%}")
    print(f"Genauigkeitsverlust: {(accuracy - faulty_accuracy) * 100:.2f} Prozentpunkte")
    print(f"Ungültige Ausgaben ohne/mit Injektion: {invalid_reference}/{invalid_faulty}")
    print(f"Geänderte Vorhersagen bei gültigen Paaren: {changed_predictions}/{valid_pairs}")
    print(f"Neu entstandene Erkennungsfehler (inkl. ungültig): {new_errors}/{correct_reference}")
    print(f"Nachgewiesene Aktivierungsinjektionen: {verified_injections}/{total}")
    print("Injektionstest: " + ("bestanden" if passed else "fehlgeschlagen"))
    return {
        "passed": passed,
        "total": total,
        "accuracy": accuracy,
        "faulty_accuracy": faulty_accuracy,
        "invalid_reference": invalid_reference,
        "invalid_faulty": invalid_faulty,
        "valid_pairs": valid_pairs,
        "changed_predictions": changed_predictions,
        "new_errors": new_errors,
        "verified_injections": verified_injections,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Testet eine Aktivierungsinjektion."
    )
    parser.add_argument(
        "--model", choices=("both", "cnn", "mlp"), default="both"
    )
    parser.add_argument(
        "--dataset", choices=("validation", "test"), default="validation"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(__file__).resolve().parents[2] / "checkpoints"

    if args.dataset == "validation":
        _, dataset = random_split(
            load_mnist(train=True),
            [50_000, 10_000],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        dataset = load_mnist(train=False)

    print(f"Datensatz: {args.dataset} | Bilder: {len(dataset)} | Gerät: {device}")

    selected_models = MODEL_CONFIGS if args.model == "both" else (args.model,)
    results = [
        run_injection(name, dataset, device, checkpoint_dir)
        for name in selected_models
    ]

    return 0 if all(result["passed"] for result in results) else 1

if __name__ == "__main__":
    raise SystemExit(main())