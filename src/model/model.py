# Convolutional Neural Network for recognising MNIST handwritten digits

import torch
from torch import nn

# === CNN-Hauptklasse ===
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,  # Eingabe-Bilder haben nur einen Eingabekanal: Graustufenwert
            out_channels=8, # Wir arbeiten mit acht lernbaren Filtern
            kernel_size=3,  # Jeder Filter betrachtet einen Bereich von 3 x 3 Werten
            padding=1,      # Padding von 1 px um Eingabebild herum, damit 3 x 3-Filtermaske auch auf die Randpixel angewendet werden kann (also die Mitte des Filters auf einem Randpixel sein kann)
        )

        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # Faltung - Filterkerne werden über das Eingabebild geschoben
        x = self.conv1(x)   # [4, 1, 28, 28] -> [4, 8, 28, 28] (je ein Ausgabekanal pro Filter)

        # ReLU auf Ausgabe -> keine negativen Werte
        x = self.relu1(x)   # [4, 8, 28, 28] -> [4, 8, 28, 28]

        # (Max-)Pooling - reduziert Datenmenge (und damit den Aufwand), indem jede Merkmalskarte (= Ausgabekanal) in vier Teile geteilt und die größten Werte übernommen werden (sodass aus 28 x 28 ein 14 x 14-Fenster wird) 
        x = self.pool1(x)   # [4, 8, 28, 28] -> [4, 8, 14, 14]
        return x


# === Test ===
if __name__ == "__main__":
    model = MNIST_CNN()

    example_images = torch.randn(4, 1, 28, 28)  # Testeingabe: Vier Bilder, ein Kanal, 28 x 28 px
    output = model(example_images)

    print("Eingabe: ", example_images.shape)
    print("Ausgabe: ", output.shape)