from pathlib import Path
import argparse
import random
import struct

# NN
import torch
from torch import nn
from pytorchfi.core import fault_injection

# lokale Module
from src.model.cnn import MNIST_CNN
from src.model.mlp import MNIST_MLP
from src.model.train import evaluate
from src.data.mnist import load_mnist


# === Vorkonfigurationen: Modellklasse, Checkpoint-Dateiname, Schichttyp, Schichtname und Position im Tensor, in den injiziert wird ===
MODEL_CONFIGS = {
    "cnn": (MNIST_CNN, "mnist_baseline.pt", nn.Conv2d, "conv1", ([0], [0], [0], [0],)), # jede Liste sollte gleich viele Elemente haben, da alle Elemente an einem Index zu einer Injektion gehöen 
    "mlp": (MNIST_MLP, "mnist_mlp_baseline.pt", nn.Linear, "hidden", ([0, 0], # Index des Layers
                                                                      [0, 10], # Ausgang am Neuron  
                                                                      [0, 14*28+14], # Eingang, 14*28+14 -> mnist-Bildmitte
                                                                      )),
}


# Diese Klasse definiert eine Funktion flip_bits, welche dazu genutzt wird, die Gewichte unseres CNNs mit bit flips zu manipulieren.
# flip_bits wird dabei an die PyTorchFI api gegeben als Funktion die zur Injektion benutzt wird.
# flip_bits wird dann für jede angegebene Injektion (Siehe pfi_model_2.declare_weight_fi Aufruf), aufgerufen.
# Alle anderen Methoden der Klasse sind Hilfsmethoden.
class bitflip(fault_injection):
    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)

    # ARGUMENTE:
    # weight ist kompletter Gewichtstensor (alle Filter der gewählten Schicht): [k, c, kH, kW]
    # index ist ein tupel mit den Koordinaten des zu verändernden Wertes (k, c, kH, kW)
    # k... index des Filters, c... channel index, kH... Kernelhöhe, kW... Kernelwidth
    # RÜCKGABE:
    # gibt den neuen Wert welcher an der festgelegten Stelle gesetzt werden soll
    def flip_bits(self, weight, index):
        orig = weight[index].item()
        injected = self.flip_bit_30(orig)
        orig_bits = self.float_to_bits(orig)
        inj_bits = self.float_to_bits(injected)
        print(f"original: {str(orig)} as bitstring: {orig_bits[0]} {orig_bits[1:9]} {orig_bits[9:]}")
        print(f"injected: {str(injected)} as bitstring: {inj_bits[0]} {inj_bits[1:9]} {inj_bits[9:]}")
        return injected

    def float_to_bits(self, float):
        [i] = struct.unpack("<I", struct.pack("<f", float))
        return f"{i:032b}"

    def flip_random_bit_in_exponent(self, float):
        flip_offset = 23
        flip_index = random.randint(0, 7)
        # flip_offset + flip_index geben einen index zwischen 23-30, was die Bits des Exponenten in einem float32 sind (VEM)
        bit_index = flip_offset + flip_index
        as_int = struct.unpack("<I", struct.pack("<f", float))[0]
        print("flipping bit at position: " + str(bit_index) + "\n")
        as_int ^= (1 << bit_index)
        return struct.unpack("<f", struct.pack("<I", as_int))[0]
    
    def flip_random_bit(self, float):
        flip_index = random.randint(0, 31)
        as_int = struct.unpack("<I", struct.pack("<f", float))[0]
        print("flipping bit at position: " + str(flip_index) + "\n")
        as_int ^= (1 << flip_index)
        return struct.unpack("<f", struct.pack("<I", as_int))[0]
    
    def flip_sign(self, float):
        as_int = struct.unpack("<I", struct.pack("<f", float))[0]
        as_int ^= (1 << 31)
        return struct.unpack("<f", struct.pack("<I", as_int))[0]

    # Bit 30 richtet am meisten Schaden an, da es das höchste Bit des Exponenten bei float32 in VEM Darstellung ist.
    # Der Wert des floats ergibt sich aus V (Bit 31) E (Bits 23-30) M (Bits 0-22) wie folgt:
    # wert = (-1)^V * (1 + M/2^23) * 2^(E - 127)
    # Somit ist das Drehen des Bits gleichzusetzen wie wenn man die Zahl mit 2^128 multiplizieren würde, (wenn es vorher 0 war)
    # oder sie mit 2^128 zu dividieren (wenn es vorher 1 war).
    def flip_bit_30(self, float):
        as_int = struct.unpack("<I", struct.pack("<f", float))[0]
        as_int ^= (1 << 30)
        return struct.unpack("<f", struct.pack("<I", as_int))[0]

        

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    parser = argparse.ArgumentParser(description="Testet eine Aktivierungsinjektion.")
    parser.add_argument("--model", choices=("cnn", "mlp"), default="cnn")
    args = parser.parse_args()
    selected_model = args.model
    model_class, checkpoint_name, layer_type, layer_name, position = MODEL_CONFIGS[selected_model]

    print(f"\n=== {selected_model.upper()} | {layer_name} | Position {position} ===")

    # === Modell aus gespeichertem Checkpoint laden ===
    project_root = Path(__file__).resolve().parents[2]
    checkpoint_path = (
        project_root / "checkpoints" / checkpoint_name
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = model_class().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  

    # Testdatensatz laden
    dataset = load_mnist(train=False)

    # === eigene Klasse, welche von injector erbt benutzen ===
    pfi_model_2 = bitflip(
                    model, # Das geladene Modell
                    batch_size=1, # ganzen Datensatz
                    input_shape=[1, 28, 28], # Keine Batchdimension -> Graustufenbild mit 28 x 28 px
                    layer_types=[layer_type], # Faltungsschichten als möglche Injektionspunkte auswählen
                    use_cuda=device.type == "cuda", # Zum Modell passendes Gerät wählen  
                )

    pfi_model_2.print_pytorchfi_layer_summary()

    # Injektion festlegen
    if selected_model == "cnn": 
        faulty_model = pfi_model_2.declare_weight_fi(
                                    function=pfi_model_2.flip_bits, # eigene Funktion festlegen
                                    layer_num=position[0], # index des Layers in den injiziert wird (also welche Conv2 Schicht)
                                    k=position[1], # index des Filters im Layers
                                    dim1=[0 for i in position[0]], # dimension des channels (wir haben nur einen, da Graustufen anstatt rgb, deshalb nur Liste mit Nullen)
                                    dim2=position[2], # Kernelhöhe des Filters
                                    dim3=position[3], # Kernelbreite des Filters
                                    )
    elif selected_model == "mlp":
        faulty_model = pfi_model_2.declare_weight_fi(
                                    function=pfi_model_2.flip_bits, # eigene Funktion festlegen
                                    layer_num=position[0], # index des Layers in den injiziert wird (also welche Linear Schicht)
                                    k=position[1], # in Position in Layer
                                    dim1=position[2], # out position im Layer
                                    dim2=[None for i in position[0]], # brauch Liste mit None (obwohl dimension hier nicht gebraucht wird), da sonst leere Liste genommen wird (wirft Error)
                                    dim3=[None for i in position[0]],
                                    )

    # Beide Modelle ausführen
    try:
        with torch.no_grad():
            accuracy = evaluate(model, dataset, device)
            faulty_accuracy = evaluate(faulty_model, dataset, device)
    finally:
        None
    
    # Ergebnisse vergleichen
    print(f"Datensatzgröße: {len(dataset)}")
    print(f"Accuracy ohne Injektion: {accuracy}")
    print(f"Accuracy mit Injektion: {faulty_accuracy}")

if __name__ == "__main__":
    main()