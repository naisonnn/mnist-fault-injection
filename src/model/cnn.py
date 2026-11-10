# Convolutional Neural Network for recognising MNIST handwritten digits

# === Neural Network-Imports ===
from torch import nn

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