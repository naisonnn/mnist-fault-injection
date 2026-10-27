# Convolutional Neural Network for recognising MNIST handwritten digits

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# ===== CNN-HAUPTKLASSE =====
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # === 1. CNN-Block ===
        self.conv1 = nn.Conv2d(
            in_channels=1,      # Eingabe-Bilder haben nur einen Eingabekanal: Graustufenwert
            out_channels=8,     # Wir arbeiten mit acht lernbaren Filtern
            kernel_size=3,      # Jeder Filter betrachtet einen Bereich von 3 x 3 Werten
            padding=1,          # Padding von 1 px um Eingabebild herum, damit 3 x 3-Filtermaske auch auf die Randpixel angewendet werden kann (also die Mitte des Filters auf einem Randpixel sein kann)
        )

        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # === 2. CNN-Block ===
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
        x = self.conv1(x)       # [4, 1, 28, 28] -> [4, 8, 28, 28] (je ein Ausgabekanal pro Filter)

        # ReLU auf Ausgabe -> keine negativen Werte
        x = self.relu1(x)       # [4, 8, 28, 28] -> [4, 8, 28, 28]

        # (Max-)Pooling - reduziert Datenmenge (und damit den Aufwand), indem jede Merkmalskarte (= Ausgabekanal) in vier Teile geteilt und die größten Werte übernommen werden (sodass aus 28 x 28 ein 14 x 14-Fenster wird) 
        x = self.pool1(x)       # [4, 8, 28, 28] -> [4, 8, 14, 14]


        # === 2. Block ===
        x = self.conv2(x)       # [4, 1, 14, 14] -> [4, 16, 14, 14]
        x = self.relu2(x)       # [4, 1, 14, 14] -> [4, 16, 14, 14]
        x = self.pool2(x)       # [4, 1, 14, 14] -> [4, 16, 7, 7]

        # === Klassifikation ===
        x = self.flatten(x)     # [4, 16, 7, 7] -> [4, 784]
        x = self.classifier(x)  # [4, 784] -> [4, 10]

        return x
    
# ===== TRAINING =====
def train_model(model, train_data, epochs=5):
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True) # trainingsdaten in 64 Gruppen teilen und mischen
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # der Optimizer aktualisiert die Gewichte des Netzes

    model.train() # setzt Modell in Trainingsmodus (setzt training flag auf True)
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad() # löscht Gradienten des vorherigen Batches
            prediction = model(images) # Vorhersage machen
            loss = loss_fn(prediction, labels) # berechnet loss aus Vorhersage und wahren Antworten
            loss.backward() # berechnet für jedes Gewicht den Gradienten des Loss
            optimizer.step() # aktualisiert alle Gewichte anhand dieser Gradienten
        print(f"Epoche {epoch+1}: letzter Loss {loss.item():.4f}")


# ===== TEST =====
def evaluate(model, test_data):
    test_loader = DataLoader(test_data, batch_size=1000)

    model.eval() # setzt training flag auf False
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            prediction = model(images) # Vorhersage, hat Form [1000, 10], 1000 wegen batch_size und 10 sind die 10 Werte der Output-Neuronen
            # jeweils pro Zeile (mit den 10 Einträgen) wird immer der Index des Eintrags mit der höchsten Zahl gespeichert
            # Dieser Index sollte jetzt gleich der Zahl sein welche auf dem Bild abgebildet ist
            predicted_numbers = prediction.argmax(dim=1)
            correct += (predicted_numbers == labels).sum().item() #sum liefert tensor mit einzigem Wert, .item() macht daraus einen integer
    return correct / len(test_data)


# ===== AUSFÜHRUNG =====
if __name__ == "__main__":
    model = MNIST_CNN()
    train_model(model, train_data)
    accuracy = evaluate(model, test_data)
    print(f"Genauigkeit: {accuracy:.2%}")
    

    # example_images = torch.randn(4, 1, 28, 28)  # Testeingabe: Vier Bilder, ein Kanal, 28 x 28 px
    # output = model(example_images)

    # print("Eingabe: ", example_images.shape)
    # print("Ausgabe: ", output.shape)

    # print("Klassenwerte des ersten Bildes: ", output[0])
    # print("Vorhergesagte Ziffern: ", output.argmax(dim=1))  # argmax liefert Position des größten Klassenwertes (= größter Wahrscheinlichkeit einer Klasse)