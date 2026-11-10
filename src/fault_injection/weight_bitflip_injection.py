from pathlib import Path
import torch
import random
import struct
from torch import nn
from pytorchfi.core import fault_injection

from src.model.cnn import MNIST_CNN
from src.model.train import evaluate
from src.data.mnist import load_mnist


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

    # Testdatensatz laden
    dataset = load_mnist(train=False)

    # === eigene Klasse, welche von injector erbt benutzen ===
    pfi_model_2 = bitflip(
                    model, # Das geladene Modell
                    batch_size=1, # ganzen Datensatz
                    input_shape=[1, 28, 28], # Keine Batchdimension -> Graustufenbild mit 28 x 28 px
                    layer_types=[nn.Conv2d], # Faltungsschichten als möglche Injektionspunkte auswählen
                    use_cuda=device.type == "cuda", # Zum Modell passendes Gerät wählen  
                )

    pfi_model_2.print_pytorchfi_layer_summary()

    # Injektion festlegen
    faulty_model = pfi_model_2.declare_weight_fi(
                                function=pfi_model_2.flip_bits, # eigene Funktion festlegen
                                layer_num=[0], # index des Layers in den injiziert wird (also welche Conv2 Schicht)
                                k=[0], # index des Filters im Layers
                                dim1=[0], # dimension des channels (wir haben nur einen, da Graustufen anstatt rgb)
                                dim2=[0], # Kernelhöhe des Filters
                                dim3=[0], # Kernelbreite des Filters
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